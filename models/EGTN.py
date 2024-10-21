import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .layers import ESLayer, ETLayer, merge_time_dim, separate_time_dim


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    return emb


class EGTN(nn.Module):
    def __init__(self, n_layers, node_dim, edge_dim, hidden_dim, time_emb_dim, act_fn,
                 learn_ref_frame, n_layers_ref, num_w, scale=1, pre_norm=False):
        super().__init__()
        self.s_modules = nn.ModuleList()
        self.t_modules = nn.ModuleList()
        self.n_layers = n_layers
        self.time_emb_dim = time_emb_dim
        self.input_linear = nn.Linear(node_dim + time_emb_dim, hidden_dim)

        self.learn_ref_frame = learn_ref_frame
        self.n_layers_ref = n_layers_ref
        self.num_w = num_w  # Should normally equal to the length of the predicted trajectory, T_f

        self.scale = scale

        # Parse activation
        if act_fn == 'silu':
            act_fn = nn.SiLU()
        else:
            raise NotImplementedError(act_fn)

        for i in range(n_layers):
            self.s_modules.append(
                ESLayer(node_dim=hidden_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, act_fn=act_fn,
                        normalize=True, pre_norm=pre_norm)
            )
            self.t_modules.append(
                ETLayer(node_dim=hidden_dim, hidden_dim=hidden_dim, act_fn=act_fn, time_emb_dim=time_emb_dim)
            )

        # Ref frame networks
        if self.learn_ref_frame:
            self.input_linear_ref = nn.Linear(node_dim, hidden_dim)
            self.s_modules_ref = nn.ModuleList()
            self.t_modules_ref = nn.ModuleList()
            for i in range(n_layers_ref):
                self.s_modules_ref.append(
                    ESLayer(node_dim=hidden_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, act_fn=act_fn,
                            normalize=True, pre_norm=pre_norm)
                )
                self.t_modules_ref.append(
                    ETLayer(node_dim=hidden_dim, hidden_dim=hidden_dim, act_fn=act_fn, time_emb_dim=time_emb_dim)
                )
            self.ws = nn.Parameter(data=torch.ones(num_w), requires_grad=True)

    def forward(self, diffusion_t, x, h, edge_index, edge_attr, batch, **model_kwargs):
        """
        :param diffusion_t: The diffusion time step, shape [BN,]
        :param x: shape [BN, 3, T]
        :param h: shape [BN, H] or [BN, H, T]
        :param edge_index: shape [2, BM]
        :param edge_attr: shape [BM, He]
        :param batch: shape [BN]
        """

        compute_ref = model_kwargs.get('compute_ref', False)
        if compute_ref:
            # Derive the reference frame based on x_given
            # Get condition mask and concat the condition frames
            cond_mask = model_kwargs.get('cond_mask', None)  # [1, 1, T]
            cond_mask = cond_mask.view(-1).bool()
            x_given = model_kwargs['x_given']
            x = x_given[..., cond_mask]  # [BN, 3, T_p]

            x = x * self.scale

            T = x.size(-1)
            if h.dim() == 2:
                h = h.unsqueeze(-1).repeat(1, 1, T)  # [BN, Hh, T_p]
            else:
                h = h[..., :T]
            h = separate_time_dim(self.input_linear_ref(merge_time_dim(h)), t=T)
            if edge_attr is not None:
                edge_attr = edge_attr.unsqueeze(-1).repeat(1, 1, T)  # [BM, He, T]

            for i in range(self.n_layers_ref):
                x, h = self.s_modules_ref[i](x, h, edge_index, edge_attr, batch, **model_kwargs)
                x, h = self.t_modules_ref[i](x, h)  # [BN, 3, T], [BN, H, T]

            # Aggregate x using normalized h
            h = h.mean(dim=1, keepdim=True)  # [BN, 1, T]
            # h[..., -1] = 1 - h[..., :-1].sum(dim=-1)
            # x_ref = (x * h).sum(dim=-1, keepdim=True)  # [BN, 3, 1]
            wt = h  # [BN, 1, T]
            ws = self.ws.view(1, 1, -1).repeat(wt.size(0), 1, 1)  # [BN, 1, S]
            wts = wt.unsqueeze(-1) * ws.unsqueeze(-2)  # [BN, 1, T, S]
            wts[:, :, -1, :] = 1 - wts[:, :, :-1, :].sum(dim=-2)  # [BN, 1, T, S]
            x_ref = (x.unsqueeze(-1) * wts).sum(dim=-2)  # [BN, 3, S]

            x_ref = x_ref / self.scale

            return x_ref

        else:
            # Get condition mask and concat the condition frames
            cond_mask = model_kwargs.get('cond_mask', None)  # [1, 1, T]
            if cond_mask is not None:
                cond_mask = cond_mask.view(-1).bool()
                x_given = model_kwargs['x_given']
                x_cond = x_given[..., cond_mask]
                x_input = x  # Record x in order to subtract it in the end for translation invariance
                x = torch.cat((x_cond, x), dim=-1)
            else:
                x_input = x

            x = x * self.scale

            T = x.size(-1)
            diffusion_t = get_timestep_embedding(diffusion_t, embedding_dim=self.time_emb_dim)  # [BN, Ht]
            diffusion_t = diffusion_t.unsqueeze(-1).repeat(1, 1, T)  # [BN, Ht, T]
            t_emb = diffusion_t

            if h.dim() == 2:
                h = h.unsqueeze(-1).repeat(1, 1, T)
            else:
                pass

            h = torch.cat((h, t_emb), dim=1)  # [BN, Hh+Ht+Ht, T]
            h = separate_time_dim(self.input_linear(merge_time_dim(h)), t=T)  # [BN, H, T]
            if edge_attr is not None:
                edge_attr = edge_attr.unsqueeze(-1).repeat(1, 1, T)  # [BM, He, T]

            for i in range(self.n_layers):
                x, h = self.s_modules[i](x, h, edge_index, edge_attr, batch, **model_kwargs)
                x, h = self.t_modules[i](x, h)

            # Clip the output through the conditional mask
            if cond_mask is not None:
                x = x[..., ~cond_mask]
                h = h[..., ~cond_mask]

            # Let x be translation invariant
            x = x - x_input

            x = x / self.scale

            return x, h


if __name__ == '__main__':
    import numpy as np

    BN = 5
    B = 2
    Hh = 16
    He = 2
    H = 32
    T = 10

    model = EGTN(n_layers=3, node_dim=Hh, edge_dim=He, hidden_dim=H, time_emb_dim=64, act_fn='silu',
                 learn_ref_frame=False, n_layers_ref=2, num_w=10, scale=1, pre_norm=False)

    batch = torch.from_numpy(np.array([0, 0, 0, 1, 1])).long()
    row = [0, 0, 1, 3]
    col = [1, 2, 2, 4]
    row = torch.from_numpy(np.array(row)).long()
    col = torch.from_numpy(np.array(col)).long()
    h = torch.rand(BN, Hh)
    x = torch.rand(BN, 3, T)
    edge_index = torch.stack((row, col), dim=0)  # [2, BM]
    BM = edge_index.size(-1)
    edge_attr = torch.rand(BM, He)

    t = torch.randint(0, 1000, size=(B,)).to(x)[batch]
    x_out, h_out = model(t, x, h, edge_index, edge_attr, batch)
    assert x_out.size() == x.size()
    assert h_out.size(0) == x.size(0)
    assert h_out.size(1) == H
    print('Test successful')

