import numpy as np
import os
import pickle
import torch
from .trajdata import TrajData
from .trajdataset import TrajDataset
from utils.misc import MD17_Transform


class MD17Traj(TrajDataset):
    molecules_to_download_files = {
        'aspirin': 'md17_aspirin.npz',
        'benzene': 'md17_benzene2017.npz',
        'ethanol': 'md17_ethanol.npz',
        'malonaldehyde': 'md17_malonaldehyde.npz',
        'naphthalene': 'md17_naphthalene.npz',
        'salicylic': 'md17_salicylic.npz',
        'toluene': 'md17_toluene.npz',
        'uracil': 'md17_uracil.npz',
    }
    _lambda = 1.6
    fc = False  # set to true will lead to better performance but slower training/inference

    split_ratio = [0.6, 0.2, 0.2]

    def __init__(self, root, molecule_name, with_h, down_sample_every, span, force_reprocess=False, force_length=None,
                 mode=None, return_index=False, project=False):
        assert molecule_name in self.molecules_to_download_files
        self.molecule_name = molecule_name
        self.force_length = force_length
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.with_h = with_h
        self.return_index = return_index
        self.x, self.v, self.z, self.edges = None, None, None, None
        self.h, self.edge_index, self.edge_attr = None, None, None
        self.n_traj = 1
        self.span = span
        self.down_sample_every = down_sample_every
        self.sample_interval = None
        self.project = project
        name = molecule_name if not with_h else molecule_name + '_h'
        super().__init__(root, name, force_reprocess)
        print(f'{name} using {len(self)} data points.')

    def processed_file(self):
        return os.path.join(self.root, self.name + '.pt')

    def preprocess_raw(self):
        data_dir = os.path.join(self.root, self.molecules_to_download_files[self.molecule_name])
        data = np.load(data_dir)

        x = data['R']
        v = x[1:] - x[:-1]
        x = x[:-1]  # [T, N, 3]
        z = data['z']

        if not self.with_h:
            x = x[:, z > 1, ...]
            v = v[:, z > 1, ...]
            z = z[z > 1]

        def d(_i, _j, _t):
            return np.sqrt(np.sum((x[_t][_i] - x[_t][_j]) ** 2))

        n = x.shape[1]

        atom_edges = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    _d = d(i, j, 0)
                    if _d < self._lambda:
                        atom_edges[i][j] = 1

        with open(self.processed_path, 'wb') as f:
            pickle.dump((x, v, z, atom_edges), f)
        print(f'Data saved to {self.processed_path}')

    def postprocess(self):
        # Convert into torch tensor
        self.x = torch.Tensor(self.data[0])  # [T, N, 3]
        self.v = torch.Tensor(self.data[1])  # [T, N, 3]
        self.z = torch.Tensor(self.data[2])  # [N]
        self.edges = torch.Tensor(self.data[3])  # [N, N]

        transform = MD17_Transform(max_atom_type=10, charge_power=2, max_hop=3, cutoff=1.6, fc=self.fc)
        h, edge_index, edge_attr = transform(self.x[0], self.z)  # h dim: max_atom_type * (charge_power + 1)
        self.h, self.edge_index, self.edge_attr = h, edge_index, edge_attr  # edge attr dim: max_hop + 1

        # Down sampling
        down_sample_idx = torch.arange(0, self.x.size(0), self.down_sample_every)
        self.x = self.x[down_sample_idx]
        self.v = self.v[down_sample_idx]

        # Apply split ratio
        tot_length = self.x.size(0)
        if self.mode == 'train':
            idx = slice(0, int(tot_length * self.split_ratio[0]))
        elif self.mode == 'val':
            idx = slice(int(tot_length * self.split_ratio[0]),
                        int(tot_length * (self.split_ratio[0] + self.split_ratio[1])))
        elif self.mode == 'test':
            idx = slice(int(tot_length * (self.split_ratio[0] + self.split_ratio[1])),
                        int(tot_length * (self.split_ratio[0] + self.split_ratio[1] + self.split_ratio[2])))
        else:
            raise NotImplementedError()
        print(self.mode, idx)
        self.x = self.x[idx]
        self.v = self.v[idx]

        _len = self.x.size(0) - self.span + 1
        _num = len(self)
        self.sample_interval = _len // _num
        print('Sample interval:', self.sample_interval)
        assert self.sample_interval >= 1
        assert (len(self) - 1) * self.sample_interval < _len

    def __len__(self):
        _num = 5000 if self.mode == 'train' else 1000
        if self.force_length is not None:
            _num = min(self.force_length, _num)
        _num = min(_num, self.x.size(0) - self.span + 1)
        return _num

    def __getitem__(self, idx):  # return a TrajData
        idx_start = idx * self.sample_interval
        idx_end = idx_start + self.span
        x = self.x[idx_start: idx_end]  # [_T, N, 3]
        v = self.v[idx_start: idx_end]  # [_T, N, 3]
        h = self.h  # [N, K]
        edge_index = self.edge_index
        edge_attr = self.edge_attr
        data = TrajData(x=x.permute(1, 2, 0), v=v.permute(1, 2, 0), h=h,
                        edge_index=edge_index, edge_attr=edge_attr)
        if self.return_index:
            data['system_id'] = torch.ones(1) * idx
        if self.project:
            x_mean = data.x.mean(dim=(0, 2), keepdim=True)
            data['x'] = data.x - x_mean
        return data


if __name__ == '__main__':
    molecule = 'aspirin'
    dataset = MD17Traj(root='data/md17', molecule_name=molecule, with_h=True, down_sample_every=10, span=30, mode='train')
    print(dataset[100].x)
    print(len(dataset))
    dataset = MD17Traj(root='data/md17', molecule_name=molecule, with_h=True, down_sample_every=10, span=30, mode='val', force_length=256)
    print(len(dataset))
    dataset = MD17Traj(root='data/md17', molecule_name=molecule, with_h=True, down_sample_every=10, span=30, mode='test')
    print(len(dataset))
    data = dataset[100]
    print(data)
