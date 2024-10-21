import torch
import torch.nn as nn
from tqdm import tqdm
import yaml
import argparse
from torch_geometric.data import DataLoader
from torch_geometric.nn.pool import global_mean_pool, global_add_pool
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torchmetrics.aggregation import CatMetric
from easydict import EasyDict
import wandb
import os
import shutil
import pickle
import sys
sys.path.append('./')

from experiments.fixddp import DistributedSamplerNoDuplicate
from datasets.nbody import NBody
from models.EGTN import EGTN
from diffusion.GeoTDM import GeoTDM, ModelMeanType, ModelVarType, LossType
from utils.misc import set_seed, gather_across_gpus

torch.multiprocessing.set_sharing_strategy('file_system')


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
    dataset_train = NBody(**config.data.train)
    dataset_val = NBody(**config.data.val)

    if world_size > 1:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSamplerNoDuplicate(dataset_val, shuffle=False, drop_last=False)
    else:
        sampler_train = None
        sampler_val = None

    dataloader_train = DataLoader(dataset_train, batch_size=config.train.batch_size // world_size,
                                  shuffle=(sampler_train is None), sampler=sampler_train, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=config.train.eval_batch_size // world_size,
                                shuffle=False, sampler=sampler_val, pin_memory=True)

    # Init model and optimizer
    denoise_network = EGTN(**config.denoise_model).to(rank)
    if world_size > 1:
        denoise_network = DistributedDataParallel(denoise_network, device_ids=[rank])
    diffusion = GeoTDM(denoise_network=denoise_network,
                       model_mean_type=ModelMeanType.EPSILON,
                       model_var_type=ModelVarType.FIXED_LARGE,
                       loss_type=LossType.MSE,
                       device=rank,
                       rescale_timesteps=False,
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

    # Start training
    num_epochs = config.train.num_epochs
    tot_step = 0

    best_val_nll, best_val_mse = 1e10, 1e10
    reduce_placeholder = CatMetric()

    if rank == 0:
        progress_bar = tqdm(total=num_epochs)

    for epoch in range(1, num_epochs + 1):
        if rank == 0:
            # print(f'Start epoch {epoch}')
            progress_bar.set_description(f"Epoch {epoch}")
        denoise_network.train()
        sampler_train.set_epoch(epoch)

        # Training

        train_loss_epoch, counter = torch.zeros(1).to(rank), torch.zeros(1).to(rank)

        for step, data in enumerate(dataloader_train):
            tot_step += 1

            data = data.to(rank)
            model_kwargs = {'h': data.h,
                            'edge_index': data.edge_index,
                            'edge_attr': data.edge_attr,
                            'batch': data.batch}
            x_start = data.x
            if diffusion.mode == 'cond':
                # Construct cond mask
                cond_mask = torch.zeros(1, 1, x_start.size(-1)).to(x_start)
                for interval in config.train.cond_mask:
                    cond_mask[..., interval[0]: interval[1]] = 1
                model_kwargs['cond_mask'] = cond_mask
                model_kwargs['x_given'] = x_start
                x_start_ = x_start[..., ~cond_mask.view(-1).bool()]
            else:
                model_kwargs['x_given'] = x_start
                x_start_ = x_start[..., :config.train.tot_len]

            training_losses = diffusion.training_losses(x_start=x_start_, t=None, model_kwargs=model_kwargs)
            loss = training_losses['loss']  # [BN]
            loss = global_mean_pool(loss, data.batch)  # [B]

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
            if rank == 0:
                print(f'Validating at epoch {epoch}')
            denoise_network.eval()
            val_nll_epoch, val_mse_epoch = torch.zeros(1).to(rank), torch.zeros(1).to(rank)
            counter = torch.zeros(1).to(rank)

            for step, data in enumerate(dataloader_val):
                data = data.to(rank)
                model_kwargs = {'h': data.h,
                                'edge_index': data.edge_index,
                                'edge_attr': data.edge_attr,
                                'batch': data.batch}
                x_start = data.x
                if diffusion.mode == 'cond':
                    # Construct cond mask
                    cond_mask = torch.zeros(1, 1, x_start.size(-1)).to(x_start)
                    for interval in config.train.cond_mask:
                        cond_mask[..., interval[0]: interval[1]] = 1
                    model_kwargs['cond_mask'] = cond_mask
                    model_kwargs['x_given'] = x_start
                    x_start_ = x_start[..., ~cond_mask.view(-1).bool()]
                else:
                    model_kwargs['x_given'] = x_start
                    x_start_ = x_start[..., :config.train.tot_len]

                val_results = diffusion.calc_bpd_loop(x_start=x_start_, model_kwargs=model_kwargs)
                total_bpd = val_results['total_bpd']  # [BN]
                mse = val_results['mse'].mean(dim=1)  # [BN, T] -> [BN]
                total_bpd = global_add_pool(total_bpd, data.batch)  # [B]
                mse = global_mean_pool(mse, data.batch)  # [B]

                val_nll_epoch += total_bpd.sum()
                val_mse_epoch += mse.sum()
                counter += total_bpd.size(0)

            val_nll_epoch = gather_across_gpus(val_nll_epoch, reduce_placeholder).sum().item()
            val_mse_epoch = gather_across_gpus(val_mse_epoch, reduce_placeholder).sum().item()
            counter = gather_across_gpus(counter, reduce_placeholder).sum().item()

            if rank == 0:
                print(f'Val counter: {counter}')
                val_nll_epoch = val_nll_epoch / counter
                val_mse_epoch = val_mse_epoch / counter
                print(f'Val nll: {val_nll_epoch}')
                wandb.log({"Val nll": val_nll_epoch}, commit=False)
                wandb.log({"Val mse": val_mse_epoch}, commit=True)

                better = False

                if val_nll_epoch < best_val_nll:
                    best_val_nll = val_nll_epoch
                    better = True
                if val_mse_epoch < best_val_mse:
                    best_val_mse = val_mse_epoch
                if better:
                    torch.save(denoise_network.state_dict(),
                               os.path.join(output_path, f'ckpt_best.pt'))

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

    # Start testing
    if config.train.final_test and diffusion.mode == 'cond':

        # Load dataset
        test_dataset = NBody(**config.data.test)
        if world_size > 1:
            sampler = DistributedSamplerNoDuplicate(test_dataset, shuffle=False, drop_last=False)
        else:
            sampler = None
        test_dataloader = DataLoader(test_dataset, batch_size=config.train.eval_batch_size // world_size,
                                     shuffle=False, sampler=sampler)

        # Load checkpoint
        model_ckpt_path = os.path.join(output_path, f'ckpt_{config.train.final_test_ckpt}.pt')
        state_dict = torch.load(model_ckpt_path)
        denoise_network.load_state_dict(state_dict)
        if rank == 0:
            print(f'Model loaded from {model_ckpt_path}')

        test_nll_epoch_all, test_mse_epoch_all = [], []
        minADE_K_all, minFDE_K_all = [], []  # distance is L2-norm
        aveADE_K_all, aveFDE_K_all = [], []  # distance is L2-norm
        system_id_all = []  # the index in the test dataset
        reduce_placeholder = CatMetric()

        for step, data in tqdm(enumerate(test_dataloader), disable=rank != 0):
            data = data.to(rank)
            model_kwargs = {'h': data.h,
                            'edge_index': data.edge_index,
                            'edge_attr': data.edge_attr,
                            'batch': data.batch}

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

            total_bpd = global_add_pool(total_bpd, data.batch)  # [B]
            mse = global_mean_pool(mse, data.batch)  # [B]
            test_nll_epoch_all.append(total_bpd)
            test_mse_epoch_all.append(mse)

            x_target = x_start[..., ~cond_mask.view(-1).bool()]
            shape_to_pred = x_target.shape  # [BN, 3, T_p]

            ADE_K, FDE_K = [], []

            # Compute traj distance
            for k in tqdm(range(config.train.K), disable=rank != 0):
                x_out = diffusion.p_sample_loop(shape=shape_to_pred, progress=False,
                                                model_kwargs=model_kwargs)  # [BN, 3, T_p]
                distance = (x_out - x_target).square().sum(dim=1).sqrt()  # [BN, T_p]
                distance = global_mean_pool(distance, data.batch)  # [B, T_p]
                ADE_K.append(distance.mean(dim=1))  # [B]
                FDE_K.append(distance[..., -1])  # [B]

            # Compute minADE, minFDE
            ADE_K = torch.stack(ADE_K, dim=-1)  # [B, K]
            FDE_K = torch.stack(FDE_K, dim=-1)  # [B, K]
            minADE_K_all.append(ADE_K.min(dim=-1).values)  # [B]
            minFDE_K_all.append(FDE_K.min(dim=-1).values)  # [B]
            aveADE_K_all.append(ADE_K.mean(dim=-1))  # [B]
            aveFDE_K_all.append(FDE_K.mean(dim=-1))  # [B]
            system_id_all.append(data.system_id)  # [B]

        # Analyze
        minADE_K_all = torch.cat(minADE_K_all, dim=0)  # [B_tot]
        minFDE_K_all = torch.cat(minFDE_K_all, dim=0)  # [B_tot]
        aveADE_K_all = torch.cat(aveADE_K_all, dim=0)  # [B_tot]
        aveFDE_K_all = torch.cat(aveFDE_K_all, dim=0)  # [B_tot]
        nll_all = torch.cat(test_nll_epoch_all, dim=0)  # [B_tot]
        system_id_all = torch.cat(system_id_all, dim=0)  # [B_tot]

        # Reduce from all gpus and compute metrics
        if world_size > 1:
            minADE_K_all = gather_across_gpus(minADE_K_all, reduce_placeholder)  # [B_tot * num_gpus]
            minFDE_K_all = gather_across_gpus(minFDE_K_all, reduce_placeholder)
            aveADE_K_all = gather_across_gpus(aveADE_K_all, reduce_placeholder)
            aveFDE_K_all = gather_across_gpus(aveFDE_K_all, reduce_placeholder)
            nll_all = gather_across_gpus(nll_all, reduce_placeholder)
            system_id_all = gather_across_gpus(system_id_all, reduce_placeholder)

        results = {
            f'minADE_{config.train.K}': minADE_K_all.mean().item(),
            f'minFDE_{config.train.K}': minFDE_K_all.mean().item(),
            f'aveADE_{config.train.K}': aveADE_K_all.mean().item(),
            f'aveFDE_{config.train.K}': aveFDE_K_all.mean().item(),
            'nll': nll_all.mean().item(),
            'system_id_range': [system_id_all.min().item(), system_id_all.max().item()]
        }

        if rank == 0:
            print(results)
            wandb.log({f'Test minADE_{config.train.K}': minADE_K_all.mean().item()}, commit=False)
            wandb.log({f'Test minFDE_{config.train.K}': minFDE_K_all.mean().item()}, commit=False)
            wandb.log({f'Test aveADE_{config.train.K}': aveADE_K_all.mean().item()}, commit=False)
            wandb.log({f'Test aveFDE_{config.train.K}': aveFDE_K_all.mean().item()}, commit=False)
            wandb.log({f'Test nll': nll_all.mean().item()}, commit=True)

            # Save
            save_path = os.path.join(output_path, 'results.pkl')
            save_results = {
                f'minADE_{config.train.K}': minADE_K_all.detach().cpu().numpy(),
                f'minFDE_{config.train.K}': minFDE_K_all.detach().cpu().numpy(),
                f'aveADE_{config.train.K}': aveADE_K_all.detach().cpu().numpy(),
                f'aveFDE_{config.train.K}': aveFDE_K_all.detach().cpu().numpy(),
                'nll': nll_all.detach().cpu().numpy(),
                'system_id': system_id_all.detach().cpu().numpy()
            }
            with open(save_path, 'wb') as f:
                pickle.dump(save_results, f)
            print(f'Results saved to {save_path}')

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        progress_bar.close()


def main():

    parser = argparse.ArgumentParser(description='GeoTDM')
    parser.add_argument('--train_yaml_file', type=str, help='path of the train yaml file',
                        default='configs/nbody_train_cond.yaml')
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

