"""
The classification, predictive, and marginal scores for evaluating the generated samples.
The metrics have been used in https://proceedings.mlr.press/v202/zhou23i.html
Code adapted from https://github.com/alexzhou907/ls4/blob/main/metrics.py
"""
import torch
from torch_geometric.loader import DataLoader
import random
from models.eqmotion_nbody import EqMotion
from tqdm import tqdm
from copy import deepcopy
import pickle
import torch.nn as nn
import argparse


class ClassificationDataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def classification_score(data):

    n_nodes = 5

    # Prepare data with labels
    all_data = []
    for i in range(len(data)):
        cur_data = data[i].to_data_list()
        n_nodes = cur_data[0].x.size(0)
        # A workaround to separate x_pred
        x_pred = data[i].x_pred
        bs = data[i].batch.max().item() + 1
        for j in range(bs):
            cur_x_pred = x_pred[data[i].batch == j]
            cur_data[j].x_pred = cur_x_pred

        all_data.extend(cur_data)
    print(all_data[0])

    all_data_pred = deepcopy(all_data)
    for i in range(len(all_data)):
        all_data[i]['flag'] = torch.zeros(1)
    for i in range(len(all_data_pred)):
        all_data_pred[i]['flag'] = torch.ones(1)
        all_data_pred[i].x = all_data_pred[i].x_pred

    all_data = all_data + all_data_pred

    random.shuffle(all_data)
    length = len(all_data)
    train_data = all_data[:int(length * 0.8)]
    test_data = all_data[int(length * 0.8):]

    # Create dataset
    dataset_train = ClassificationDataset(train_data)
    dataset_test = ClassificationDataset(test_data)
    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=128, shuffle=False)

    # Prepare model
    frames_in = 20
    frames_out = 1
    rank = 0
    model = EqMotion(in_node_nf=frames_in, in_edge_nf=2, hidden_nf=16, in_channel=frames_in, hid_channel=16,
                     out_channel=frames_out, device=rank, n_layers=1, recurrent=True)
    model = model.to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # Start
    n_iters = 100
    pbar = tqdm(range(n_iters))
    for i in range(n_iters):
        for data in dataloader_train:
            data = data.to(rank)
            optimizer.zero_grad()

            # n_nodes = 5
            loc = data.x
            loc = loc.view(-1, n_nodes, loc.size(-2), loc.size(-1)).permute(0, 1, 3, 2)  # [B, N, T, 3]
            vel = torch.zeros_like(loc)
            nodes = data.h  # [BN, H]

            if nodes.size(-1) > 1:
                nodes = torch.zeros_like(nodes)[..., :1]
            nodes = nodes.view(-1, n_nodes, nodes.size(-1)).repeat(1, 1, frames_in)  # [B, N, H]

            loc_pred, h_pred = model(nodes, loc.detach(), vel, edge_attr=None)  # [B, N, T_f, 3], [B, N, H]
            # print(len(h_pred))
            # print(h_pred[0].shape)
            pred = h_pred.mean(dim=(1, 2))  # [B]
            label = data.flag

            loss = torch.nn.BCEWithLogitsLoss()(pred, label)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            test_loss = 0
            counter = 0
            for ind, data in enumerate(dataloader_test):
                data = data.to(rank)
                # n_nodes = 5
                loc = data.x
                loc = loc.view(-1, n_nodes, loc.size(-2), loc.size(-1)).permute(0, 1, 3, 2)  # [B, N, T, 3]
                vel = torch.zeros_like(loc)
                # vel = vel * constant
                nodes = data.h  # [BN, H]
                if nodes.size(-1) > 1:
                    nodes = torch.zeros_like(nodes)[..., :1]
                nodes = nodes.view(-1, n_nodes, nodes.size(-1)).repeat(1, 1, frames_in)  # [B, N, H]

                # nodes = torch.sqrt(torch.sum(vel ** 2, dim=-1)).detach()
                loc_pred, h_pred = model(nodes, loc.detach(), vel, edge_attr=None)  # [B, N, T_f, 3], [B, N, H]
                # print(len(h_pred))
                pred = h_pred.mean(dim=(1, 2))  # [B]
                label = data.flag

                counter += pred.size(0)

                loss = torch.nn.BCEWithLogitsLoss(reduction='none')(pred, label).detach().cpu()
                test_loss += loss.sum().item()

            final_loss = test_loss / counter

            pbar.set_description(f'Epoch {i} Test loss: {final_loss}')
            pbar.update(1)

    return final_loss


def predictive_score(data, mode='s2r'):

    n_nodes = 5

    # Prepare data
    all_data = []
    for i in range(len(data)):
        cur_data = data[i].to_data_list()
        n_nodes = cur_data[0].x.size(0)
        # A workaround to separate x_pred
        x_pred = data[i].x_pred
        bs = data[i].batch.max().item() + 1
        for j in range(bs):
            cur_x_pred = x_pred[data[i].batch == j]
            cur_data[j].x_pred = cur_x_pred

        all_data.extend(cur_data)
    print(all_data[0])

    all_data_pred = deepcopy(all_data)
    for i in range(len(all_data_pred)):
        all_data_pred[i].x = all_data_pred[i].x_pred

    # Create dataset
    if mode == 's2r':
        train_data = all_data_pred
        test_data = all_data
    else:
        train_data = all_data
        test_data = all_data_pred
    dataset_train = ClassificationDataset(train_data)
    dataset_test = ClassificationDataset(test_data)
    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=128, shuffle=False)

    # Prepare model
    frames_in = 10
    frames_out = 10
    rank = 0
    model = EqMotion(in_node_nf=frames_in, in_edge_nf=2, hidden_nf=16, in_channel=frames_in, hid_channel=16,
                     out_channel=frames_out, device=rank, n_layers=1, recurrent=True)
    model = model.to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # Start
    n_iters = 100
    pbar = tqdm(range(n_iters))
    for i in range(n_iters):
        for data in dataloader_train:
            data = data.to(rank)
            optimizer.zero_grad()

            # n_nodes = 5
            loc = data.x
            loc = loc.view(-1, n_nodes, loc.size(-2), loc.size(-1)).permute(0, 1, 3, 2)  # [B, N, T, 3]
            loc_end = loc[:, :, frames_in:, :]
            loc = loc[:, :, :frames_in, :]
            vel = torch.zeros_like(loc)
            nodes = data.h  # [BN, H]
            if nodes.size(-1) > 1:
                nodes = torch.zeros_like(nodes)[..., :1]
            nodes = nodes.view(-1, n_nodes, nodes.size(-1)).repeat(1, 1, frames_in)  # [B, N, H]

            loc_pred, h_pred = model(nodes, loc.detach(), vel, edge_attr=None)  # [B, N, T_f, 3], [B, N, H]
            loss = torch.nn.MSELoss()(loc_pred, loc_end)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            test_loss = 0
            counter = 0
            for ind, data in enumerate(dataloader_test):
                data = data.to(rank)
                # n_nodes = 5
                loc = data.x
                loc = loc.view(-1, n_nodes, loc.size(-2), loc.size(-1)).permute(0, 1, 3, 2)  # [B, N, T, 3]
                loc_end = loc[:, :, frames_in:, :]
                loc = loc[:, :, :frames_in, :]
                vel = torch.zeros_like(loc)
                # vel = vel * constant
                nodes = data.h  # [BN, H]
                if nodes.size(-1) > 1:
                    nodes = torch.zeros_like(nodes)[..., :1]
                nodes = nodes.view(-1, n_nodes, nodes.size(-1)).repeat(1, 1, frames_in)  # [B, N, H]

                # nodes = torch.sqrt(torch.sum(vel ** 2, dim=-1)).detach()
                loc_pred, h_pred = model(nodes, loc.detach(), vel, edge_attr=None)  # [B, N, T_f, 3], [B, N, H]
                # print(len(h_pred))
                loss = torch.nn.MSELoss(reduction='none')(loc_pred, loc_end)  # [B, N, T_f, 3]
                loss = loss.mean(dim=(1, 2, 3)).detach().cpu()

                counter += loss.size(0)
                test_loss += loss.sum().item()

            final_loss = test_loss / counter

            pbar.set_description(f'Epoch {i} Test loss: {final_loss}')
            pbar.update(1)

    return final_loss


class Loss(nn.Module):
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)


def histogram_torch(x, n_bins, density=True):
    a, b = x.min().item(), x.max().item()
    delta = (b - a) / n_bins
    bins = torch.arange(a, b + 1e-8, step=delta)
    count = torch.histc(x, n_bins).float()
    if density:
        count = count / delta / float(x.shape[0] * x.shape[1])
    return count, bins


class HistoLoss(Loss):
    def __init__(self, x_real, n_bins, **kwargs):
        super(HistoLoss, self).__init__(**kwargs)
        self.densities = list()
        self.locs = list()
        self.deltas = list()
        for i in range(x_real.shape[2]):
            x_i = x_real[..., i].reshape(-1, 1)
            d, b = histogram_torch(x_i, n_bins, density=True)
            self.densities.append(nn.Parameter(d).to(x_real.device))
            delta = b[1:2] - b[:1]
            loc = 0.5 * (b[1:] + b[:-1])
            self.locs.append(loc)
            self.deltas.append(delta)

    def compute(self, x_fake):  # [B, T, D]
        loss = list()

        def relu(x):
            return x * (x >= 0.).float()

        for i in range(x_fake.shape[2]):
            loc = self.locs[i].view(1, -1).to(x_fake.device)
            x_i = x_fake[:, :, i].contiguous().view(-1, 1).repeat(1, loc.shape[1])
            dist = torch.abs(x_i - loc)
            counter = (relu(self.deltas[i].to(x_fake.device) / 2. - dist) > 0.).float()
            density = counter.mean(0) / self.deltas[i].to(x_fake.device)
            abs_metric = torch.abs(density - self.densities[i].to(x_fake.device))
            loss.append(torch.mean(abs_metric, 0))
        loss_componentwise = torch.stack(loss)
        return loss_componentwise


def marginal_score(data, t):
    # Prepare data
    all_data = []
    for i in range(len(data)):
        cur_data = data[i].to_data_list()
        # A workaround to separate x_pred
        x_pred = data[i].x_pred
        bs = data[i].batch.max().item() + 1
        for j in range(bs):
            cur_x_pred = x_pred[data[i].batch == j]
            cur_data[j].x_pred = cur_x_pred
        all_data.extend(cur_data)
    print(all_data[0])

    all_data_pred = deepcopy(all_data)
    for i in range(len(all_data_pred)):
        all_data_pred[i].x = all_data_pred[i].x_pred

    n_bins = 50

    for i in range(t):
        x_real = torch.cat([all_data[j].x[..., i].unsqueeze(1) for j in range(len(all_data))], dim=0)  # [BN, 3]
        x_fake = torch.cat([all_data_pred[j].x[..., i].unsqueeze(1) for j in range(len(all_data_pred))], dim=0)
        loss = HistoLoss(x_real=x_real, n_bins=n_bins, name='marginal_loss')(x_fake).item()
        print(i, loss)

    # Get average
    x_real = torch.cat([all_data[j].x.permute(0, 2, 1) for j in range(len(all_data))], dim=0)  # [BN, T, 3]
    x_fake = torch.cat([all_data_pred[j].x.permute(0, 2, 1) for j in range(len(all_data_pred))], dim=0)  # [BN, T, 3]
    loss = HistoLoss(x_real=x_real, n_bins=n_bins, name='marginal_loss')(x_fake).item()
    return loss


def marginal_score_chem(data):
    # Prepare data
    all_data = []
    for i in range(len(data)):
        cur_data = data[i].to_data_list()
        # A workaround to separate x_pred
        x_pred = data[i].x_pred
        bs = data[i].batch.max().item() + 1
        for j in range(bs):
            cur_x_pred = x_pred[data[i].batch == j]
            cur_data[j].x_pred = cur_x_pred

        all_data.extend(cur_data)

    print(len(all_data))
    print(all_data[0])

    all_data_pred = deepcopy(all_data)
    for i in range(len(all_data_pred)):
        all_data_pred[i].x = all_data_pred[i].x_pred

    # Transform pos into chemical bond lengths
    x_ref = all_data[0].x[..., 0]  # [N, 3]
    all_edges = []
    cnt = 0
    for i in range(x_ref.size(0)):
        for j in range(i + 1, x_ref.size(0)):
            d = (x_ref[i] - x_ref[j]).square().sum().sqrt().item()
            if d < 1.6:
                all_edges.append((i, j))
                cnt += 1
    print('edge num identified:', cnt)
    assert cnt in [21, 12, 8, 19, 16, 15, 12]
    n_bins = 50

    all_loss = []

    for i in range(len(all_edges)):
        ei, ej = all_edges[i]
        all_d_real = [(all_data[j].x[ei] - all_data[j].x[ej]).square().sum(dim=0).sqrt() for j in range(len(all_data))]  # [T]
        all_d_real = torch.stack(all_d_real, dim=0).unsqueeze(-1)  # [B, T, 1]

        all_d_fake = [(all_data_pred[j].x[ei] - all_data_pred[j].x[ej]).square().sum(dim=0).sqrt() for j in range(len(all_data_pred))]  # [T]
        all_d_fake = torch.stack(all_d_fake, dim=0).unsqueeze(-1)  # [B, T, 1]

        print(all_d_real.max(), all_d_real.min())
        print(all_d_fake.max(), all_d_fake.min())
        print('*' * 8)

        loss = HistoLoss(x_real=all_d_real, n_bins=n_bins, name='marginal_loss')(all_d_fake).item()
        print(i, all_edges[i], loss)
        all_loss.append(loss)

    ave_score = sum(all_loss) / len(all_loss)
    return ave_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GeoTDM')
    parser.add_argument('--path', type=str, help='path of the result pkl file',
                        default='outputs/md17_aspirin_GeoTDM_uncond/samples.pkl')
    parser.add_argument('--chem', action='store_true', help='whether to evaluate marginal on chem bond')

    args = parser.parse_args()

    path = args.path

    with open(path, 'rb') as f:
        data = pickle.load(f)
    c_score = classification_score(data)
    print(c_score)
    p_score = predictive_score(data, mode='s2r')
    print(p_score)
    if args.chem:
        m_score = marginal_score_chem(data)
    else:
        m_score = marginal_score(data, t=20)

    print('Summary:')
    print(f'Marginal score: {m_score}')
    print(f'Classification score: {c_score}')
    print(f'Prediction score: {p_score}')

