import torch
from torch_geometric.loader import DataLoader
import pickle
import os
import shutil
import yaml
import argparse
from easydict import EasyDict
import sys
sys.path.append('./')

from datasets.nbody import NBody
from models.EGTN import EGTN
from diffusion.GeoTDM import GeoTDM, ModelMeanType, ModelVarType, LossType
from utils.misc import set_seed

parser = argparse.ArgumentParser()
parser.add_argument('--eval_yaml_file', type=str, default='configs/md17_sampling.yaml')
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()

eval_yaml_file = args.eval_yaml_file
device = args.device

# Load args
with open(eval_yaml_file, 'r') as f:
    params = yaml.safe_load(f)
config = EasyDict(params)
cond = config.eval.cond
train_output_path = os.path.join(config.eval.output_base_path, config.eval.train_exp_name)
print(f'Train output path: {train_output_path}')

set_seed(config.eval.seed)

if config.eval.model == 'GeoTDM':
    if cond:
        train_yaml_file = os.path.join(train_output_path, 'nbody_train_cond.yaml')
    else:
        train_yaml_file = os.path.join(train_output_path, 'nbody_train_uncond.yaml')
else:
    raise NotImplementedError()

with open(train_yaml_file, 'r') as f:
    train_params = yaml.safe_load(f)
train_config = EasyDict(train_params)
if config.eval.eval_exp_name is None:
    config.eval.eval_exp_name = config.eval.train_exp_name + '_eval'
eval_output_path = os.path.join(config.eval.output_base_path, config.eval.eval_exp_name)
print(f'Eval output path: {eval_output_path}')
if not os.path.exists(eval_output_path):
    os.makedirs(eval_output_path)
shutil.copy(eval_yaml_file, eval_output_path)

# Overwrite model configs from training config
if config.eval.model == 'GeoTDM':
    config.denoise_model = train_config.denoise_model
    config.diffusion = train_config.diffusion
    # Overwrite diffusion timesteps for sampling
    if config.eval.sampling_timesteps is not None:
        config.diffusion.num_timesteps = config.eval.sampling_timesteps
# Overwrite cond_mask
if cond:
    if config.eval.cond_mask is None:
        config.eval.cond_mask = train_config.train.cond_mask

# Load dataset
dataset = NBody(**config.data)
dataloader = DataLoader(dataset, batch_size=config.eval.batch_size, shuffle=False)

if config.eval.model == 'GeoTDM':
    # Init model and optimizer
    denoise_network = EGTN(**config.denoise_model).to(device)
else:
    raise NotImplementedError()

# Load checkpoint
model_ckpt_path = os.path.join(train_output_path, f'ckpt_{config.eval.model_ckpt}.pt')
state_dict = torch.load(model_ckpt_path)
try:
    denoise_network.load_state_dict(state_dict)
except:
    state_dict = {k[7:]: v for k, v in state_dict.items()}
    # denoise_network = DistributedDataParallel(denoise_network, device_ids=[0])
    denoise_network.load_state_dict(state_dict)
print(f'Model loaded from {model_ckpt_path}')

if config.eval.model == 'GeoTDM':
    diffusion = GeoTDM(denoise_network=denoise_network,
                       model_mean_type=ModelMeanType.EPSILON,
                       model_var_type=ModelVarType.FIXED_LARGE,
                       loss_type=LossType.MSE,
                       device=device,
                       rescale_timesteps=False,
                       **config.diffusion)

denoise_network.eval()

all_data = []

for step, data in enumerate(dataloader):
    data = data.to(device)
    model_kwargs = {'h': data.h,
                    'edge_index': data.edge_index,
                    'edge_attr': data.edge_attr,
                    'batch': data.batch
                    }

    x_start = data.x

    if cond:
        # Create temporal inpainting mask, 1 to keep the entries unchanged, 0 to modify it by diffusion
        cond_mask = torch.zeros(1, 1, x_start.size(-1)).to(x_start)
        for interval in config.eval.cond_mask:
            cond_mask[..., interval[0]: interval[1]] = 1
        model_kwargs['cond_mask'] = cond_mask
        shape_to_pred = x_start[..., ~cond_mask.view(-1).bool()].shape
    else:
        x_start_ = x_start[..., :train_config.train.tot_len]
        shape_to_pred = x_start_.shape
        data.x = x_start_

    model_kwargs['x_given'] = x_start

    if config.eval.model == 'GeoTDM':
        x_out = diffusion.p_sample_loop(shape=shape_to_pred, progress=True, model_kwargs=model_kwargs)
    else:
        raise NotImplementedError()

    if cond:
        x_out = torch.cat((x_start[..., cond_mask.view(-1).bool()], x_out), dim=-1)

    data['x_pred'] = x_out.detach()

    all_data.append(data.cpu())

    # break  # break here to only get a few samples


samples_save_path = os.path.join(eval_output_path, 'samples.pkl')
with open(samples_save_path, 'wb') as f:
    pickle.dump(all_data, f)
print(f'Samples saved to {samples_save_path}')





