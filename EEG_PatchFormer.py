import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import os.path as osp
import pickle

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Patcher(nn.Module):
    def __init__(self, patch_size, stride, in_chan, out_dim):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride)
        self.to_out = nn.Sequential(
            nn.Linear(int(in_chan*patch_size[0]*patch_size[1]), out_dim),
            nn.GELU()
        )
        self.to_patch = Rearrange("b l n -> b n l")

    def forward(self, x):
        # x: b, k, c, l
        x = self.unfold(x)
        x = self.to_patch(x)
        x = self.to_out(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for mixer, ff in self.layers:
            x_mix = mixer(x)
            x = ff(x_mix) + x
        return x


class PatchFormer(nn.Module):
    def temporal_learner(self, in_chan, out_chan, kernel):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1), padding=self.get_padding(kernel=kernel[-1])),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_chan),
            nn.AvgPool2d((1, 4), (1, 4))
        )

    def __init__(self, num_classes, input_size, sampling_rate, num_T, patch_time, patch_step, dim_head, depth, heads,
                 dropout_rate, idx_graph):
        # input_size: EEG frequency x channel x datapoint
        super(PatchFormer, self).__init__()
        self.idx = idx_graph
        self.window = [0.5, 0.25, 0.125]
        self.channel = input_size[1]
        self.brain_area = len(self.idx)

        # by setting the convolutional kernel being (1,length) and the stride being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.temporal_cnn = self.temporal_learner(input_size[0], num_T,
                                                  (1, int(self.window[0] * sampling_rate +1)))
        self.OneXOneConv = nn.Sequential(
            nn.Conv2d(num_T, num_T, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(num_T),
            nn.LeakyReLU(),
            nn.AvgPool2d((1, 2), (1, 2)))

        self.global_cnn = nn.Sequential(
            nn.Conv2d(num_T, num_T, kernel_size=(self.channel, 1), stride=(1, 1)),
            nn.BatchNorm2d(num_T),
            nn.LeakyReLU()
        )

        # diag(W) to assign a weight to each local areas
        size = self.get_size_temporal(input_size)
        self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.channel, size[-1]),
                                                requires_grad=True)
        nn.init.xavier_uniform_(self.local_filter_weight)
        self.local_filter_bias = nn.Parameter(torch.zeros((1, self.channel, 1), dtype=torch.float32),
                                              requires_grad=True)

        # aggregate function
        self.aggregate = Aggregator(self.idx)

        patch_chan = 1
        self.to_patch = Patcher(patch_size=(patch_chan, patch_time), stride=(1, patch_step),
                                in_chan=num_T, out_dim=dim_head)

        self.transformer = Transformer(
            dim=dim_head, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dim=dim_head, dropout=dropout_rate,
        )

        num_patch = int((self.brain_area + 1) * ((size[-1] // num_T - patch_time) // patch_step + 1))
        # learn the global network of networks
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(num_patch*dim_head), num_classes))

    def forward(self, x):
        # x: batch, chan, time
        x = torch.unsqueeze(x, dim=1)  # (batch, 1, chan, time)
        out = self.temporal_cnn(x)
        out = self.OneXOneConv(out)  # b, k, c, l
        out_global_branch = out  # b, k, c, l
        b, k, c, l = out_global_branch.size()
        out_global = self.global_cnn(out_global_branch)  # b, k, 1, l
        out = rearrange(out, 'b k c l -> b c (k l)')
        out = self.local_filter_fun(out, self.local_filter_weight)
        out = self.aggregate.forward(out)  # b, g, (k l)
        out_local = rearrange(out, 'b g (k l) -> b k g l', k=k, l=l)  # b, k, g, l
        out = torch.cat((out_global, out_local), dim=-2)   # b, k, g+1, l
        out = self.to_patch(out)    # b, n, h
        out = self.transformer(out)   # b, n, h
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

    def get_size_temporal(self, input_size):
        # input_size: frequency x channel x data point
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        out = self.temporal_cnn(data)
        out = self.OneXOneConv(out)
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        size = out.size()
        return size

    def local_filter_fun(self, x, w):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)
        return x

    def get_padding(self, kernel):
        return (0, int(0.5 * (kernel - 1)))


class Aggregator():

    def __init__(self, idx_area):
        # chan_in_area: a list of the number of channels within each area
        self.chan_in_area = idx_area
        self.idx = self.get_idx(idx_area)
        self.area = len(idx_area)

    def forward(self, x):
        # x: batch x channel x data
        data = []
        for i, area in enumerate(range(self.area)):
            if i < self.area - 1:
                data.append(self.aggr_fun(x[:, self.idx[i]:self.idx[i + 1], :], dim=1))
            else:
                data.append(self.aggr_fun(x[:, self.idx[i]:, :], dim=1))
        return torch.stack(data, dim=1)

    def get_idx(self, chan_in_area):
        idx = [0] + chan_in_area
        idx_ = [0]
        for i in idx:
            idx_.append(idx_[-1] + i)
        return idx_[1:]

    def aggr_fun(self, x, dim):
        # return torch.max(x, dim=dim).values
        return torch.mean(x, dim=dim)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # you can tune patch_time and patch_step to control the sliding window length and step
    # when patch_step = patch_time, it means we use non-overlapped patches

    original_order = ['Fp1', 'AFF5', 'AFz', 'F1', 'FC5', 'FC1', 'T7', 'C3', 'Cz', 'CP5', 'CP1', 'P7', 'P3',
                      'Pz', 'POz', 'O1', 'Fp2', 'AFF6', 'F2', 'FC2', 'FC6', 'C4', 'T8', 'CP2', 'CP6', 'P4',
                      'P8', 'O2']

    graph_general = [['Fp1', 'Fp2'], ['AFF5', 'AFz', 'AFF6'], ['F1', 'F2'],
                    ['FC5', 'FC1', 'FC2', 'FC6'], ['C3', 'Cz', 'C4'], ['CP5', 'CP1', 'CP2', 'CP6'],
                    ['P7', 'P3', 'Pz', 'P4', 'P8'], ['POz'], ['O1', 'O2'],
                    ['T7'], ['T8']]

    graph_idx = graph_general  # The general graph definition.
    idx = []
    num_chan_local_graph = []
    for i in range(len(graph_idx)):
        num_chan_local_graph.append(len(graph_idx[i]))
        for chan in graph_idx[i]:
            idx.append(original_order.index(chan))

    data = torch.randn(1, 28, 800)  # (batch_size=1, EEG_channel=28, data_points=800)
    data = data[:, idx, :]  # (batch_size=1, EEG_channel=28, data_points=800)

    net = PatchFormer(
        num_classes=2, input_size=(1, 28, 800), sampling_rate=200, num_T=32, patch_time=20, patch_step=5,
        dim_head=32, depth=4, heads=32,
        dropout_rate=0.5, idx_graph=num_chan_local_graph)
    print(net)
    print(count_parameters(net))

    out = net(data)
