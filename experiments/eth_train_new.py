import torch
import torch.nn as nn
from tqdm import tqdm
import yaml
import argparse
from torch_geometric.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torchmetrics.aggregation import CatMetric
from easydict import EasyDict
import wandb
import os
import shutil
import sys
sys.path.append('./')

from experiments.fixddp import DistributedSamplerNoDuplicate
from datasets.eth_new import ETHNew
from models.EGTN import EGTN
from diffusion.GeoTDM import GeoTDM, ModelMeanType, ModelVarType, LossType
from utils.misc import set_seed, gather_across_gpus

from torch_kmeans import KMeans

torch.multiprocessing.set_sharing_strategy('file_system')


def lr_decay(optimizer, lr_now, gamma):
    lr_new = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
    return lr_new


def run(rank, world_size, args):

    # Load args
    yaml_file = args.train_yaml_file
    with open(yaml_file, 'r') as f:
        params = yaml.safe_load(f)
    config = EasyDict(params)

    # Save args yaml file
    output_path = os.path.join(config.train.output_base_path, config.train.exp_name)
    if rank == 0:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        shutil.copy(yaml_file, output_path)

    set_seed(config.train.seed)

    # Load dataset
    dataset_train = ETHNew(return_index=False, **config.data.train)

    # Load dataset
    test_dataset = ETHNew(return_index=True, **config.data.test)
    if world_size > 1:
        sampler = DistributedSamplerNoDuplicate(test_dataset, shuffle=False, drop_last=False)
    else:
        sampler = None
    test_dataloader = DataLoader(test_dataset, batch_size=config.train.eval_batch_size // world_size, shuffle=False,
                                 sampler=sampler)


    if world_size > 1:
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_train = None

    dataloader_train = DataLoader(dataset_train, batch_size=config.train.batch_size // world_size,
                                  shuffle=(sampler_train is None), sampler=sampler_train, pin_memory=True)

    # Init model and optimizer
    denoise_network = EGTN(**config.denoise_model).to(rank)

    if world_size > 1:
        denoise_network = DistributedDataParallel(denoise_network, device_ids=[rank], find_unused_parameters=False)
    diffusion = GeoTDM(denoise_network=denoise_network,
                       model_mean_type=ModelMeanType.EPSILON,
                       model_var_type=ModelVarType.FIXED_LARGE,
                       loss_type=LossType.MSE,
                       device=rank,
                       rescale_timesteps=True,
                       **config.diffusion)

    optimizer = torch.optim.Adam(denoise_network.parameters(), lr=config.train.lr)

    if rank == 0:
        # Wandb config
        if config.wandb.no_wandb:
            mode = 'disabled'
        else:
            mode = 'online'
        kwargs = {'entity': config.wandb.wandb_usr, 'name': config.train.exp_name, 'project': config.wandb.project,
                  'config': params, 'settings': wandb.Settings(_disable_stats=True), 'mode': mode}
        wandb.init(**kwargs)
        wandb.save('*.txt')

    # Refer to SVAE paper, FPC
    post_process = config.train.cluster
    kmeans_fn = KMeans(n_clusters=20)

    # Start training
    num_epochs = config.train.num_epochs
    tot_step = 0

    best_fde = 1e10
    best_ade = 1e10
    reduce_placeholder = CatMetric()

    if rank == 0:
        progress_bar = tqdm(total=num_epochs)

    lr_now = config.train.lr

    for epoch in range(1, num_epochs + 1):
        if rank == 0:
            # print(f'Start epoch {epoch}')
            progress_bar.set_description(f"Epoch {epoch}")
        denoise_network.train()
        if world_size > 1:
            sampler_train.set_epoch(epoch)

        if epoch % config.train.lr_decay_every == 0 and epoch > 0:
            lr_now = lr_decay(optimizer, lr_now, gamma=0.8)

        # Training

        train_loss_epoch, counter = torch.zeros(1).to(rank), torch.zeros(1).to(rank)

        for step, data in enumerate(dataloader_train):
            tot_step += 1

            data = data.to(rank)
            model_kwargs = {'h': data.h,
                            'v': data.v,
                            'edge_index': data.edge_index,
                            'edge_attr': data.edge_attr,
                            'batch': data.batch,
                            'num': data.num
                            }
            x_start = data.x
            # Construct cond mask
            cond_mask = torch.zeros(1, 1, x_start.size(-1)).to(x_start)
            for interval in config.train.cond_mask:
                cond_mask[..., interval[0]: interval[1]] = 1
            model_kwargs['cond_mask'] = cond_mask
            model_kwargs['x_given'] = x_start
            x_start_ = x_start[..., ~cond_mask.view(-1).bool()]
            training_losses = diffusion.training_losses(x_start=x_start_, t=None, model_kwargs=model_kwargs)
            loss = training_losses['loss']  # [BN]
            loss = loss[data.select_index]

            if world_size > 1:
                step_loss_synced = gather_across_gpus(loss, reduce_placeholder).mean().item()
            else:
                step_loss_synced = loss.mean().item()
            if rank == 0 and tot_step % config.train.log_every_step == 0:
                wandb.log({"Step train loss": step_loss_synced}, commit=True, step=tot_step)
                logs = {"loss": step_loss_synced, "step": tot_step}
                progress_bar.set_postfix(**logs)

            train_loss_epoch = train_loss_epoch + loss.sum()
            counter = counter + loss.size(0)

            loss = loss.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(denoise_network.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        train_loss_epoch = gather_across_gpus(train_loss_epoch, reduce_placeholder).sum().item()
        counter = gather_across_gpus(counter, reduce_placeholder).sum().item()

        if rank == 0:
            wandb.log({"Epoch train loss": train_loss_epoch / counter}, commit=True)

        # Eval on validation set
        if epoch % config.train.eval_every_epoch == 0 and epoch != 0:
            # Start testing
            denoise_network.eval()
            if config.train.final_test:

                test_nll_epoch_all_node, test_mse_epoch_all_node = [], []
                minADE_K_all_node, minFDE_K_all_node = [], []  # distance is L2-norm
                if post_process:
                    minADE_K_all_node1, minFDE_K_all_node1 = [], []  # distance is L2-norm
                system_id_all = []  # the index in the test dataset
                reduce_placeholder = CatMetric()

                for step, data in tqdm(enumerate(test_dataloader), disable=rank != 0):
                    data = data.to(rank)
                    model_kwargs = {'h': data.h,
                                    'v': data.v,
                                    'edge_index': data.edge_index,
                                    'edge_attr': data.edge_attr,
                                    'batch': data.batch,
                                    'num': data.num
                                    }

                    x_start = data.x

                    # Create temporal inpainting mask, 1 to keep the entries unchanged, 0 to modify it by diffusion
                    cond_mask = torch.zeros(1, 1, x_start.size(-1)).to(x_start)
                    for interval in config.train.cond_mask:
                        cond_mask[..., interval[0]: interval[1]] = 1
                    model_kwargs['cond_mask'] = cond_mask
                    model_kwargs['x_given'] = x_start

                    x_start_ = x_start[..., ~cond_mask.view(-1).bool()]
                    val_results = diffusion.calc_bpd_loop(x_start=x_start_, model_kwargs=model_kwargs)
                    total_bpd = val_results['total_bpd']  # [BN]
                    mse = val_results['mse'].mean(dim=1)  # [BN, T] -> [BN]

                    total_bpd = total_bpd[data.select_index]
                    mse = mse[data.select_index]

                    # Should be node-wise average, as per previous papers, instead of graph-wise average
                    test_nll_epoch_all_node.append(total_bpd)
                    test_mse_epoch_all_node.append(mse)

                    x_target = x_start[..., ~cond_mask.view(-1).bool()]
                    shape_to_pred = x_target.shape  # [BN, 3, T_p]

                    ADE_K_node, FDE_K_node = [], []

                    if post_process:
                        all_final_frames = []
                        all_traj = []
                        all_target = []

                    # Compute traj distance
                    for k in tqdm(range(config.train.K), disable=rank != 0):
                        x_out = diffusion.p_sample_loop(shape=shape_to_pred, progress=False,
                                                        model_kwargs=model_kwargs)  # [BN, 3, T_p]
                        # x_out = torch.cat((x_start[..., cond_mask.view(-1).bool()], x_out), dim=-1)
                        if post_process:
                            all_final_frames.append(x_out[data.select_index][..., -1])  # [B, 3]
                            all_traj.append(x_out[data.select_index])  # [B, 3, T_p]
                            all_target.append(x_target[data.select_index])  # [B, 3, T_p]

                        distance = (x_out - x_target).square().sum(dim=1).sqrt()  # [BN, T_p]

                        distance = distance[data.select_index]  # [B, T_p] only select node 0

                        ADE_K_node.append(distance.mean(dim=1))  # [B]
                        FDE_K_node.append(distance[..., -1])  # [B]

                    if post_process:
                        all_final_frames = torch.stack(all_final_frames, dim=1)  # [B, K, 3]
                        res_centers = kmeans_fn(all_final_frames).centers  # [B, K_means, 3]
                        dis = (all_final_frames[:, :, None, :] - res_centers[:, None, :, :]).square().sum(dim=-1)
                        # [B, K, K_means]
                        index = dis.argmin(dim=1)  # [B, K_means]
                        all_traj = torch.stack(all_traj, dim=1)  # [B, K, 3, T_p]
                        selected_traj = torch.stack([all_traj[cur_i][index[cur_i]] for cur_i in range(all_traj.size(0))], dim=0)
                        # selected_traj = all_traj[index]  # [B, K_means, 3, T_p]
                        all_target = all_target[0]  # [B, 3, T_p]
                        all_error = (selected_traj - all_target.unsqueeze(1)).square().sum(dim=-2).sqrt()  # [B, K_means, T_p]
                        all_error_ave = all_error.mean(dim=-1)
                        all_error_final = all_error[..., -1]  # [B, K_means]
                        minADE_K_all_node1.append(all_error_ave.min(dim=1).values)
                        minFDE_K_all_node1.append(all_error_final.min(dim=1).values)

                    # Compute minADE, minFDE
                    system_id_all.append(data.system_id)  # [B]

                    ADE_K_node = torch.stack(ADE_K_node, dim=-1)  # [BN, K]
                    FDE_K_node = torch.stack(FDE_K_node, dim=-1)  # [BN, K]
                    minADE_K_all_node.append(ADE_K_node.min(dim=-1).values)  # [BN]
                    minFDE_K_all_node.append(FDE_K_node.min(dim=-1).values)  # [BN]

                # Analyze
                system_id_all = torch.cat(system_id_all, dim=0)  # [B_tot]

                minADE_K_all_node = torch.cat(minADE_K_all_node, dim=0) * config.data.test.traj_scale  # [B_tot]
                minFDE_K_all_node = torch.cat(minFDE_K_all_node, dim=0) * config.data.test.traj_scale  # [B_tot]
                nll_all_node = torch.cat(test_nll_epoch_all_node, dim=0)  # [B_tot]
                eps_mse_all_node = torch.cat(test_mse_epoch_all_node, dim=0) * config.data.test.traj_scale  # [B_tot]

                if post_process:
                    minADE_K_all_node1 = torch.cat(minADE_K_all_node1, dim=0) * config.data.test.traj_scale
                    minFDE_K_all_node1 = torch.cat(minFDE_K_all_node1, dim=0) * config.data.test.traj_scale

                # Reduce from all gpus and compute metrics
                if world_size > 1:
                    system_id_all = gather_across_gpus(system_id_all, reduce_placeholder)

                    minADE_K_all_node = gather_across_gpus(minADE_K_all_node, reduce_placeholder)  # [B_tot * num_gpus]
                    minFDE_K_all_node = gather_across_gpus(minFDE_K_all_node, reduce_placeholder)
                    nll_all_node = gather_across_gpus(nll_all_node, reduce_placeholder)
                    eps_mse_all_node = gather_across_gpus(eps_mse_all_node, reduce_placeholder)

                    if post_process:
                        minADE_K_all_node1 = gather_across_gpus(minADE_K_all_node1, reduce_placeholder)
                        minFDE_K_all_node1 = gather_across_gpus(minFDE_K_all_node1, reduce_placeholder)

                results = {
                    f'minADE_node': minADE_K_all_node.mean().item(),
                    f'minFDE_node': minFDE_K_all_node.mean().item(),
                    'nll_node': nll_all_node.mean().item(),
                    'eps_mse_node': eps_mse_all_node.mean().item(),
                    'system_id_range': [system_id_all.min().item(), system_id_all.max().item()]
                }
                if post_process:
                    results[f'minADE_node'] = minADE_K_all_node1.mean().item()
                    results[f'minFDE_node'] = minFDE_K_all_node1.mean().item()

                if rank == 0:
                    wandb.log({f'test_minADE': results[f'minADE_node']}, commit=False)
                    wandb.log({f'test_minFDE': results[f'minFDE_node']}, commit=False)
                    wandb.log({'test_nll': nll_all_node.mean().item()}, commit=True)

                if rank == 0:
                    print(results)
                    print('Counter:', system_id_all.size())
                    fde = results[f'minFDE_node']
                    ade = results[f'minADE_node']
                    if fde < best_fde:
                        best_fde = fde
                        best_ade = ade
                        print('Better Model!')
                        torch.save(denoise_network.state_dict(),
                                   os.path.join(output_path, f'ckpt_best.pt'))
                    print(f'current best fde {best_fde}; ade {best_ade}')

        # Save model
        if rank == 0 and config.train.save_model:
            if epoch % config.train.save_every_epoch == 0:
                torch.save(denoise_network.state_dict(),
                           os.path.join(output_path, f'ckpt_{epoch}.pt'))
            torch.save(denoise_network.state_dict(),
                       os.path.join(output_path, f'ckpt_last.pt'))

        if world_size > 1:
            dist.barrier()
        if rank == 0:
            progress_bar.update(1)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        progress_bar.close()


def main():

    parser = argparse.ArgumentParser(description='GeoTDM')
    parser.add_argument('--train_yaml_file', type=str, help='path of the train yaml file',
                        default='configs/eth_train_new.yaml')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    print(args)

    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')

    if world_size > 1:
        dist.init_process_group('nccl', rank=args.local_rank, world_size=world_size)
    run(args.local_rank, world_size, args)


if __name__ == '__main__':
    main()

