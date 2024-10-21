"""
EqMotion model, originally implemented in https://arxiv.org/abs/2303.10876
Code adapted from https://github.com/MediaBrain-SJTU/EqMotion/tree/main/n_body_system
The model is used as a surrogate model for evaluating the unconditionally generated trajectories.
See experiments/scores.py for how the model is used in evaluation.
"""
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F


class Feature_learning_layer(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, input_c, hidden_c, output_c, edges_in_d=0, nodes_att_dim=0,
                 act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False, norm_diff=False, tanh=False,
                 input_reasoning=False, category_num=2):
        super(Feature_learning_layer, self).__init__()
        self.norm_diff = norm_diff

        self.coord_vel = nn.Linear(hidden_c, hidden_c, bias=False)
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        self.hidden_c = hidden_c
        edge_coords_nf = hidden_c
        self.hidden_nf = hidden_nf

        one_coord_weight = False
        if one_coord_weight:
            layer = nn.Linear(hidden_nf, 1, bias=False)
        else:
            layer = nn.Linear(hidden_nf, hidden_c, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.clamp = False
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1)) * 3
        self.coord_mlp = nn.Sequential(*coord_mlp)

        self.tao = 0.2
        self.category_num = category_num
        self.input_reasoning = input_reasoning

        if input_reasoning:
            self.edge_mlp = nn.Sequential(
                nn.Linear(input_edge + edge_coords_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn)
            self.category_mlp = []
            for i in range(category_num):
                self.category_mlp.append(nn.Sequential(
                    nn.Linear(input_edge + edge_coords_nf, hidden_nf),
                    act_fn,
                    nn.Linear(hidden_nf, hidden_c),
                    act_fn))
            self.category_mlp = nn.ModuleList(self.category_mlp)

            self.factor_mlp = nn.Sequential(
                nn.Linear(hidden_c, hidden_c),
                act_fn,
                nn.Linear(hidden_c, hidden_c),
                act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        self.node_att_mlp = nn.Sequential(
            nn.Linear(hidden_c, hidden_c),
            act_fn,
            nn.Linear(hidden_c, 1))

        self.add_non_linear = True
        if self.add_non_linear:
            self.layer_q = nn.Linear(hidden_c, hidden_c, bias=False)
            self.layer_k = nn.Linear(hidden_c, hidden_c, bias=False)

        self.add_inner_agent_attention = True
        if self.add_inner_agent_attention:
            self.mlp_q = nn.Sequential(
                nn.Linear(hidden_nf, int(hidden_c)),
                act_fn)

    def edge_model(self, h, coord, edge_attr=None):
        batch_size, agent_num, channels = coord.shape[0], coord.shape[1], coord.shape[2]
        h1 = h[:, :, None, :].repeat(1, 1, agent_num, 1)
        h2 = h[:, None, :, :].repeat(1, agent_num, 1, 1)
        coord_diff = coord[:, :, None, :, :] - coord[:, None, :, :, :]
        coord_dist = torch.norm(coord_diff, dim=-1)
        edge_feat = torch.cat([h1, h2, coord_dist], dim=-1)
        edge_feat = self.edge_mlp(edge_feat)
        return edge_feat, coord_diff  # (B,N,N,D)

    def aggregate_coord(self, coord, edge_feat, coord_diff):
        factors = self.coord_mlp(edge_feat).unsqueeze(-1)
        neighbor_effect = torch.sum(factors * coord_diff, dim=2)
        coord = coord + neighbor_effect
        return coord

    def aggregate_coord_reasoning(self, coord, edge_feat, coord_diff, category, h):
        batch_size, agent_num, channels = coord.shape[0], coord.shape[1], coord.shape[2]
        h1 = h[:, :, None, :].repeat(1, 1, agent_num, 1)
        h2 = h[:, None, :, :].repeat(1, agent_num, 1, 1)
        coord_dist = torch.norm(coord_diff, dim=-1)
        edge_h = torch.cat([h1, h2, coord_dist], dim=-1)
        factors = torch.zeros(batch_size, agent_num, agent_num, channels).type_as(coord)
        for i in range(self.category_num):
            factors += (category[:, :, :, i:i + 1] * self.category_mlp[i](edge_h))
        factors = self.factor_mlp(factors)

        factors = factors.unsqueeze(-1)
        neighbor_effect = torch.sum(factors * coord_diff, dim=2)
        coord = coord + neighbor_effect
        return coord

    def node_model(self, x, edge_feat):
        batch_size, agent_num = edge_feat.shape[0], edge_feat.shape[1]
        mask = (torch.ones((agent_num, agent_num)) - torch.eye(agent_num)).type_as(edge_feat)
        mask = mask[None, :, :, None].repeat(batch_size, 1, 1, 1)
        aggregated_edge = torch.sum(mask * edge_feat, dim=2)
        out = self.node_mlp(torch.cat([x, aggregated_edge], dim=-1))

        if self.recurrent:
            out = x + out
        return out

    def inner_agent_attention(self, coord, h):
        att = self.mlp_q(h).unsqueeze(-1)
        v = coord - torch.mean(coord, dim=(1, 2), keepdim=True)
        out = att * v
        apply_res = True
        if apply_res:
            out = out + coord
        return out

    def non_linear(self, coord):
        coord_mean = torch.mean(coord, dim=(1, 2), keepdim=True)
        coord = coord - coord_mean
        q = self.layer_q(coord.transpose(2, 3)).transpose(2, 3)
        k = self.layer_k(coord.transpose(2, 3)).transpose(2, 3)
        product = torch.matmul(q.unsqueeze(-2), k.unsqueeze(-1)).squeeze(-1)  # (B,N,C,1)
        mask = (product >= 0).float()  # (B,N,C,1)
        EPS = 1e-4
        k_norm_sq = torch.sum(k * k, dim=-1, keepdim=True)  # (B,N,C,1)
        coord = mask * q + (1 - mask) * (q - (product / (k_norm_sq + EPS)) * k)
        coord = coord + coord_mean
        return coord

    def forward(self, h, coord, vel, edge_attr=None, node_attr=None, category=None):
        edge_feat, coord_diff = self.edge_model(h, coord, edge_attr)

        if self.add_inner_agent_attention:
            coord = self.inner_agent_attention(coord, h)

        if self.input_reasoning:
            coord = self.aggregate_coord_reasoning(coord, edge_feat, coord_diff, category, h)
        else:
            coord = self.aggregate_coord(coord, edge_feat, coord_diff)

        coord += self.coord_vel(vel.transpose(2, 3)).transpose(2, 3)

        if self.add_non_linear:
            coord = self.non_linear(coord)

        h = self.node_model(h, edge_feat)

        return h, coord, category


class EqMotion(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, in_channel, hid_channel, out_channel, device='cpu',
                 act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, recurrent=False, norm_diff=False, tanh=False):
        super(EqMotion, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        self.embedding = nn.Linear(in_node_nf, int(self.hidden_nf / 2))
        self.embedding2 = nn.Linear(in_node_nf, int(self.hidden_nf / 2))
        # self.embedding2 = nn.Linear(in_node_nf, int(self.hidden_nf))

        self.coord_trans = nn.Linear(in_channel, int(hid_channel), bias=False)
        self.vel_trans = nn.Linear(in_channel, int(hid_channel), bias=False)

        self.predict_head = nn.Linear(hid_channel, out_channel, bias=False)
        self.apply_dct = True
        self.validate_reasoning = False
        self.in_channel = in_channel
        self.out_channel = out_channel

        category_num = 2
        self.category_num = category_num
        self.tao = 1

        self.given_category = False
        if not self.given_category:
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_nf * 2 + hid_channel * 2, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn)

            self.coord_mlp = nn.Sequential(
                nn.Linear(hid_channel * 2, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hid_channel * 2),
                act_fn)

            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_nf + int(1 * hidden_nf), hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn)
            # self.gumbel_noise = self.sample_gumbel((category_num), eps=1e-10).cuda()

            self.category_mlp = nn.Sequential(
                nn.Linear(hidden_nf * 2 + hid_channel * 2, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, category_num),
                act_fn)

        for i in range(0, n_layers):
            if i == n_layers - 1:
                self.add_module("gcl_%d" % i,
                                Feature_learning_layer(self.hidden_nf, self.hidden_nf, self.hidden_nf, in_channel,
                                                       hid_channel, out_channel, edges_in_d=in_edge_nf, act_fn=act_fn,
                                                       coords_weight=coords_weight, recurrent=recurrent,
                                                       norm_diff=norm_diff, tanh=tanh, input_reasoning=True))
            else:
                self.add_module("gcl_%d" % i,
                                Feature_learning_layer(self.hidden_nf, self.hidden_nf, self.hidden_nf, in_channel,
                                                       hid_channel, out_channel, edges_in_d=in_edge_nf, act_fn=act_fn,
                                                       coords_weight=coords_weight, recurrent=recurrent,
                                                       norm_diff=norm_diff, tanh=tanh, input_reasoning=True))

        self.to(self.device)

    def get_dct_matrix(self, N, x):
        dct_m = np.eye(N)
        for k in np.arange(N):
            for i in np.arange(N):
                w = np.sqrt(2 / N)
                if k == 0:
                    w = np.sqrt(1 / N)
                dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
        idct_m = np.linalg.inv(dct_m)
        dct_m = torch.from_numpy(dct_m).type_as(x)
        idct_m = torch.from_numpy(idct_m).type_as(x)
        return dct_m, idct_m

    def transform_edge_attr(self, edge_attr):
        edge_attr = (edge_attr / 2) + 1
        interaction_category = F.one_hot(edge_attr.long(), num_classes=self.category_num)
        return interaction_category

    def calc_category(self, h, coord):
        import torch.nn.functional as F
        batch_size, agent_num, channels = coord.shape[0], coord.shape[1], coord.shape[2]
        h1 = h[:, :, None, :].repeat(1, 1, agent_num, 1)
        h2 = h[:, None, :, :].repeat(1, agent_num, 1, 1)
        coord_diff = coord[:, :, None, :, :] - coord[:, None, :, :, :]
        coord_dist = torch.norm(coord_diff, dim=-1)
        coord_dist = self.coord_mlp(coord_dist)
        edge_feat_input = torch.cat([h1, h2, coord_dist], dim=-1)
        # edge_feat_input = coord_dist
        edge_feat = self.edge_mlp(edge_feat_input)
        mask = (torch.ones((agent_num, agent_num)) - torch.eye(agent_num)).type_as(edge_feat)
        mask = mask[None, :, :, None].repeat(batch_size, 1, 1, 1)
        node_new = self.node_mlp(torch.cat([h, torch.sum(mask * edge_feat, dim=2)], dim=-1))
        node_new1 = node_new[:, :, None, :].repeat(1, 1, agent_num, 1)
        node_new2 = node_new[:, None, :, :].repeat(1, agent_num, 1, 1)

        edge_feat_input_new = torch.cat([node_new1, node_new2, coord_dist], dim=-1)
        interaction_category = F.softmax(self.category_mlp(edge_feat_input_new) / self.tao, dim=-1)

        return interaction_category

    def forward(self, h, x, vel, edge_attr=None):  # x shape: [B, N, T_p, 3]
        # hinit = torch.zeros()
        vel_pre = torch.zeros_like(vel)
        vel_pre[:, :, 1:] = vel[:, :, :-1]
        vel_pre[:, :, 0] = vel[:, :, 0]
        EPS = 1e-6
        vel_cosangle = torch.sum(vel_pre * vel, dim=-1) / (
                    (torch.norm(vel_pre, dim=-1) + EPS) * (torch.norm(vel, dim=-1) + EPS))

        vel_angle = torch.acos(torch.clamp(vel_cosangle, -1, 1))

        batch_size, agent_num, length = x.shape[0], x.shape[1], x.shape[2]
        if self.apply_dct:
            x_center = torch.mean(x, dim=(1, 2), keepdim=True)
            x = x - x_center
            dct_m, _ = self.get_dct_matrix(self.in_channel, x)
            _, idct_m = self.get_dct_matrix(self.out_channel, x)
            dct_m = dct_m[None, None, :, :].repeat(batch_size, agent_num, 1, 1)
            idct_m = idct_m[None, None, :, :].repeat(batch_size, agent_num, 1, 1)
            x = torch.matmul(dct_m, x)
            vel = torch.matmul(dct_m, vel)

        h = self.embedding(h)
        vel_angle_embedding = self.embedding2(vel_angle)
        h = torch.cat([h, vel_angle_embedding], dim=-1)

        x_mean = torch.mean(torch.mean(x, dim=-2, keepdim=True), dim=-3, keepdim=True)
        x = self.coord_trans((x - x_mean).transpose(2, 3)).transpose(2, 3) + x_mean
        vel = self.vel_trans(vel.transpose(2, 3)).transpose(2, 3)
        x_cat = torch.cat([x, vel], dim=-2)
        cagegory_per_layer = []
        if self.given_category:
            category = self.transform_edge_attr(edge_attr)
        else:
            category = self.calc_category(h, x_cat)

        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, x, vel, edge_attr=edge_attr, category=category)
            cagegory_per_layer.append(category)
        x_mean = torch.mean(torch.mean(x, dim=-2, keepdim=True), dim=-3, keepdim=True)
        x = self.predict_head((x - x_mean).transpose(2, 3)).transpose(2, 3) + x_mean
        if self.apply_dct:
            x = torch.matmul(idct_m, x)
            x = x + x_center
        if self.validate_reasoning:
            return x, cagegory_per_layer
        else:
            return x, h
