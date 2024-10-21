from torch_geometric.data.data import Data
from torch_geometric.typing import *
import torch
from typing import Any, Sequence

IndexType = Union[int, slice, Tensor, Sequence]


class TrajData(Data):

    def __init__(self, h: OptTensor = None, edge_index: OptTensor = None, edge_attr: OptTensor = None,
                 x: Union[OptTensor, List[OptTensor]] = None,
                 v: Union[OptTensor, List[OptTensor]] = None,
                 t: Union[List[float], float, OptTensor] = None, **kwargs):
        """
        The trajectory data class. TODO: Currently only work on graph with static invariant node feature.
        :param h: The node feature in shape [N, H]
        :param edge_index: The edge index in shape [2, M]
        :param edge_attr: The edge attribute in shape [M, He]
        :param x: The position in shape [N, 3 ,T]
        :param v: The velocity in shape [N, 3, T]
        :param t: The time stamp in shape [T]
        :param kwargs:
        """
        # Formatting position, velocity, and timestamp
        if x is not None:
            x = _time_dim_to_tensor(x)
        if v is not None:
            v = _time_dim_to_tensor(v)
        if t is not None:
            if isinstance(t, List):
                t = torch.Tensor(t)
            if t.dim() == 1:
                t = t.unsqueeze(0)  # add an additional dimension for mini-batching
        super().__init__(h=h, edge_index=edge_index, edge_attr=edge_attr, x=x, v=v, t=t, **kwargs)
        # self._check_time_alignment()

    def n_windows(self, span: int) -> int:
        return len(self) - span + 1

    @property
    def v(self):
        return self['v'] if 'v' in self._store else None

    @property
    def state(self):
        return torch.stack((self.x, self.v), dim=-1)

    def __len__(self):
        return self.x.size(-1)

    def __getitem__(self, key: Union[str, IndexType]) -> Any:
        if isinstance(key, str):  # key index
            return self._store[key]
        else:  # time index
            return self.at(key)

    def at(self, t: IndexType) -> Any:
        if isinstance(t, tuple):
            t = list(t)
        # Special treatment for the time dimension
        if self.t is not None:
            reduced_t = self.t[..., t]
            if reduced_t.dim() == 1:
                reduced_t = reduced_t.unsqueeze(-1)
        else:
            reduced_t = None
        ret = TrajData(
            h=self.h, edge_index=self.edge_index, edge_attr=self.edge_attr,
            x=self.x[..., t] if self.x is not None else None,
            v=self.v[..., t] if self.v is not None else None,
            t=reduced_t,
        )
        # copy other attributes
        for key in self._store:
            if key not in ret._store:
                ret._store[key] = self[key]
        return ret

    def cut(self, t_idx: IndexType):
        if self.x is not None:
            self._store['x'] = self.x[..., t_idx]
        if self.v is not None:
            self._store['v'] = self.v[..., t_idx]
        if self.t is not None:
            self._store['t'] = self.t[..., t_idx]

    def roll(self, next_x: OptTensor = None, next_v: OptTensor = None, next_t: OptTensor = None):
        if next_x is not None:
            delta_t = next_x.size(-1)
            self['x'] = torch.cat((self.x[..., delta_t:], next_x), dim=-1)
        if next_v is not None:
            delta_t = next_v.size(-1)
            self['v'] = torch.cat((self.vel[..., delta_t:], next_v), dim=-1)
        if next_t is not None:
            delta_t = next_t.size(-1)
            self['t'] = torch.cat((self.t[..., delta_t:], next_t), dim=-1)

    def snapshot(self, reduce_x: Optional[str] = 'last',
                 reduce_v: Optional[str] = 'last',
                 reduce_t: Optional[str] = 'last') -> Data:
        # TODO: Deal with NoneType pos, vel, and t
        if reduce_x == 'last':
            x = self.x[..., -1]
        elif reduce_x == 'concat':
            x = self.x.view(self.x.size(0), -1)
        else:
            x = self.x
        if reduce_v == 'last':
            v = self.v[..., -1]
        elif reduce_v == 'concat':
            v = self.v.view(self.v.size(0), -1)
        else:
            v = self.v
        if reduce_t == 'last':
            t = self.t[..., -1]
        else:
            t = self.t
        ret = Data(h=self.h, edge_index=self.edge_index, edge_attr=self.edge_attr, x=x, v=v, t=t)
        # copy other attributes
        for key in self._store:
            if key not in ret._store:
                ret._store[key] = self[key]
        return ret

    def _check_time_alignment(self):
        if self.x is not None:
            assert len(self) == self.x.size(-1)
        if self.v is not None:
            assert len(self) == self.v.size(-1)


def _time_dim_to_tensor(traj: Union[OptTensor, List[OptTensor]]) -> Tensor:
    if isinstance(traj, List):
        traj = torch.stack(traj, dim=-1)
        return traj
    elif isinstance(traj, Tensor):
        if len(traj.shape) == 2:
            traj = traj.unsqueeze(-1)
            return traj
        elif len(traj.shape) == 3:
            return traj
        else:
            raise NotImplementedError(f"Unknown traj shape: {traj.shape}")
    else:
        raise NotImplementedError(f"Unknown traj type: {traj.type}")
