import torch
import numpy as np
import random
from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as F


def set_seed(seed):
    # Fix seed
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def gather_across_gpus(metric, placeholder):
    placeholder.update(metric)
    ret = placeholder.compute()
    placeholder.reset()
    return ret


class MD17_Transform(object):
    """
    Featurization technique, adapted from https://arxiv.org/abs/2105.03902 and
    https://proceedings.neurips.cc/paper_files/paper/2021/file/a45a1d12ee0fb7f1f872ab91da18f899-Paper.pdf
    Useful for generative modeling of 3D molecules.
    """
    def __init__(self, max_atom_type, charge_power, max_hop, cutoff, fc):
        self.max_atom_type = max_atom_type
        self.charge_power = charge_power
        self.max_hop = max_hop
        self.cutoff = cutoff
        self.fc = fc

    def binarize(self, x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(self, adj, order):

        adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device),
                    self.binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]
        for i in range(2, order + 1):
            adj_mats.append(self.binarize(adj_mats[i - 1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)
        for i in range(1, order + 1):
            order_mat += (adj_mats[i] - adj_mats[i - 1]) * i
        return order_mat

    def gen_fully_connected_with_hop(self, pos):
        nodes = pos.shape[0]
        adj = torch.norm(pos.unsqueeze(0) - pos.unsqueeze(1), p=2, dim=-1)  # n * n
        adj = (adj <= self.cutoff) & (~torch.eye(nodes).bool())
        adj_order = self.get_higher_order_adj_matrix(adj.long(), self.max_hop)
        if self.fc:
            fc = 1 - torch.eye(pos.shape[0], dtype=torch.long)
            ans = adj_order + fc
            edge_index, edge_type = dense_to_sparse(ans)
        else:
            edge_index, edge_type = dense_to_sparse(adj_order)
        return edge_index, edge_type - 1

    def gen_atom_onehot(self, atom_type):
        if self.charge_power == -1:
            return atom_type
        else:
            one_hot = F.one_hot(atom_type.long(), self.max_atom_type)
            charge_tensor = (atom_type.unsqueeze(-1) / self.max_atom_type).pow(
                torch.arange(self.charge_power + 1., dtype=torch.float32))
            charge_tensor = charge_tensor.view(atom_type.shape + (1, self.charge_power + 1))
            atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(atom_type.shape + (-1,))
            return atom_scalars

    def __call__(self, x, h):
        h = self.gen_atom_onehot(h)
        edge_index, edge_type = self.gen_fully_connected_with_hop(x)
        edge_attr = F.one_hot(edge_type, self.max_hop + 1)
        return h, edge_index, edge_attr

