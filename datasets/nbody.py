from typing import Union, List, Tuple
import numpy as np
import torch
import os.path as osp
import pickle
from torch_geometric.utils import remove_self_loops
from .trajdataset import TrajDataset
from .trajdata import TrajData


class NBody(TrajDataset):
    def __init__(self, root, name, suffix, span, force_reprocess=False, force_length=None, return_index=False,
                 obs_noise_scale=0, project=False):
        self.suffix = suffix
        self.span = span
        self.force_length = force_length
        self.return_index = return_index
        self.x, self.v, self.charge, self.edges, self.edge_attr = None, None, None, None, None
        self.noise = None
        self.project = project
        self.obs_noise_scale = obs_noise_scale
        super().__init__(root=root, name=name, force_reprocess=force_reprocess)
        print(f'{name} using {len(self)} data points.')

    def raw_files(self) -> Union[str, List[str], Tuple]:
        names = [osp.join(self.root, 'loc_' + self.suffix + '.npy'),
                 osp.join(self.root, 'vel_' + self.suffix + '.npy'),
                 osp.join(self.root, 'edges_' + self.suffix + '.npy'),
                 osp.join(self.root, 'charges_' + self.suffix + '.npy')]
        return names

    def processed_file(self):
        return osp.join(self.root, self.name + '.pt')

    def preprocess_raw(self):
        loc, vel, edges, charges = [np.load(file) for file in self.raw_files()]
        if 'gravity' in self.name:
            loc, vel = torch.Tensor(loc), torch.Tensor(vel)
        else:
            loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        n_nodes = loc.size(2)
        # loc = loc[:self.max_samples, :, :, :]  # limit number of samples
        # charges = charges[0:self.max_samples]

        if 'gravity' in self.name:
            edges = np.ones(shape=(loc.shape[0], n_nodes, n_nodes))

        edge_attr = []

        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(
            2)  # swap n_nodes <--> batch_size and add nf dimension
        x = torch.Tensor(loc)  # [n_traj, T, N, 3]
        v = torch.Tensor(vel)  # [n_traj, T, N, 3]
        edge_attr = torch.Tensor(edge_attr)  # [n_traj, N, N]
        edges = torch.from_numpy(np.array(edges))  # [n_traj, N, N]
        charges = torch.Tensor(charges)  # [n_traj, N, 1]

        with open(self.processed_path, 'wb') as f:
            pickle.dump((x, v, charges, edges, edge_attr), f)
        print(f'Data saved to {self.processed_path}')

    def postprocess(self):
        # Convert into torch tensor
        self.x = torch.Tensor(self.data[0])  # [T, N, 3]
        self.v = torch.Tensor(self.data[1])  # [T, N, 3]
        self.charge = torch.Tensor(self.data[2])  # [N, 1]
        self.edges = torch.Tensor(self.data[3])  # [N, N]
        self.edge_attr = torch.Tensor(self.data[4])  # [N, N]
        # Load noise matrix
        if self.obs_noise_scale > 0:
            with open(osp.join(self.root, self.name + '_noise.pt'), 'rb') as f:
                noise = pickle.load(f)
            self.noise = torch.Tensor(noise)

    def __len__(self):
        return self.x.size(0) if self.force_length is None else min(self.force_length, self.x.size(0))  # [n_traj]

    def __getitem__(self, idx):  # return a TrajData
        x = self.x[idx]  # [_T, N, 3]
        if self.obs_noise_scale > 0:
            noise = self.noise[idx]
            x = x + noise * self.obs_noise_scale
        v = self.v[idx]  # [_T, N, 3]
        h = self.charge[idx]  # [N, 1]
        num_nodes = x.size(1)
        row = torch.arange(num_nodes, dtype=torch.long)
        col = torch.arange(num_nodes, dtype=torch.long)
        row = row.view(-1, 1).repeat(1, num_nodes).view(-1)
        col = col.repeat(num_nodes)
        edge_index = torch.stack([row, col], dim=0)
        # remove self loop
        if 'spring' in self.name:
            edge_index = remove_self_loops(edge_index)[0]
            edge_attr = self.edge_attr[idx].view(-1, 1)
        elif 'gravity' in self.name:
            edge_index = remove_self_loops(edge_index)[0]
            edge_attr = torch.zeros(edge_index.shape[1]).view(-1, 1)
        else:
            edge_index = remove_self_loops(edge_index)[0]
            edge_attr = h[edge_index[0]] * h[edge_index[1]]  # This is the same as using self.edge_attr[idx]

        if self.span is not None:  # Cut the trajectory to Traj[0:span]
            x = x[:self.span]
            v = v[:self.span]

        data = TrajData(x=x.permute(1, 2, 0), v=v.permute(1, 2, 0), h=h,
                        edge_index=edge_index, edge_attr=edge_attr)
        if self.return_index:
            data['system_id'] = torch.ones(1) * idx
        if self.project:
            x_mean = data.x.mean(dim=(0, 2), keepdim=True)
            data['x'] = data.x - x_mean

        return data


if __name__ == '__main__':
    dataset = NBody(root='datasets/datagen', name='gravity_valid', suffix='valid_gravity10_initvel1', span=30)
    # dataset = NBody(root='datasets/datagen', name='spring_test', suffix='test_springs5_initvel1', span=30)
    print(len(dataset))
    print(dataset[10])

