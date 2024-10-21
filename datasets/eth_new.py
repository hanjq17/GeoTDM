import torch
import numpy as np
import torch_geometric.data.collate

from .trajdata import TrajData


class ETHNew(object):
    def __init__(self, dataset, past_frames, future_frames, traj_scale, phase, return_index=False):
        # file_dir = 'eth_ucy/processed_data_diverse/'
        file_dir = 'data/eth/'
        if phase == 'training':
            data_file_path = file_dir + dataset +'_data_train.npy'
            num_file_path = file_dir + dataset +'_num_train.npy'
        elif phase == 'testing':
            data_file_path = file_dir + dataset +'_data_test.npy'
            num_file_path = file_dir + dataset +'_num_test.npy'
        all_data = np.load(data_file_path)
        all_num = np.load(num_file_path)
        self.all_data = torch.Tensor(all_data)
        self.all_num = torch.Tensor(all_num)
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.traj_scale = traj_scale
        self.return_index = return_index

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, item):
        all_seq = self.all_data[item] / self.traj_scale  # [N, T, D]
        num = self.all_num[item].long()

        all_seq = all_seq[:num]
        x = all_seq.permute(0, 2, 1)  # [N, D, T]
        x_in = x[..., :self.past_frames]
        v_in = torch.zeros_like(x_in)
        v_in[..., 1:] = x_in[..., 1:] - x_in[..., :-1]
        v_in[..., 0] = v_in[..., 1]  # [N, D, T_p]
        h = torch.zeros(x.size(0), 2, x.size(-1))
        h[:, 0, :self.past_frames] = 1
        v_in_norm = torch.norm(v_in, p=2, dim=1, keepdim=True)
        h[:, 1:, :self.past_frames] = v_in_norm
        h[:, 1:, self.past_frames:] = v_in_norm[..., -1:].repeat(1, 1, self.future_frames)
        num_nodes = x.size(0)
        row = torch.arange(num_nodes, dtype=torch.long)
        col = torch.arange(num_nodes, dtype=torch.long)
        row = row.view(-1, 1).repeat(1, num_nodes).view(-1)
        col = col.repeat(num_nodes)
        edge_index = torch.stack([row, col], dim=0)
        data = TrajData(h=h, x=x, edge_index=edge_index, v=v_in)

        select_index = torch.zeros(1)
        data['select_index'] = select_index.long()

        data['num'] = num

        if self.return_index:
            data['system_id'] = torch.ones(1) * item

        return data


if __name__ == '__main__':
    dataset = ETHNew(dataset='eth', past_frames=8, future_frames=12, traj_scale=1., phase='testing', return_index=True)
    print(len(dataset))
    print(dataset[10])
    temp = torch_geometric.data.collate.collate(cls=dataset[0].__class__, data_list=[dataset[i] for i in range(0, 100)])[0]
    print(temp)
    print(temp.num)
    print(temp.edge_attr)

