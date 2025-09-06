# https://github.com/Mr-TalhaIlyas/SegNext/tree/a73f344163213ccf9850693a6779edfb94597051

import torch
import torch.nn.functional as F
import torch.nn as nn
import yaml, math, os
from functools import partial
from torch.nn.modules.batchnorm import _BatchNorm
import queue
import collections
import threading
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast
with open('./config_torch.segnext.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

import warnings
from torch.nn.modules.utils import _pair as to_2tuple










class _MatrixDecomposition2DBase(nn.Module):
    '''
    Base class for furhter implementing the NMF, VQ or CD as in paper

    https://arxiv.org/pdf/2109.04553.pdf
    this script only has NMF as it has best performance for semantic segmentation
    as mentioned in paper
    D (dictionery) in paper is bases
    C (codes) in paper is coef here
    '''

    def __init__(self, config):
        super().__init__()

        self.spatial = config['SPATIAL']

        self.S = config['MD_S']
        self.D = config['MD_D']
        self.R = config['MD_R']

        self.train_steps = config['TRAIN_STEPS']
        self.eval_steps = config['EVAL_STEPS']

        self.inv_t = config['INV_T']
        self.eta = config['Eta']

        self.rand_init = config['RAND_INIT']

    def _bild_bases(self, B, S, D, R):
        raise NotImplementedError

    def local_setp(self, x, bases, coef):
        raise NotImplementedError

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        # here: N = HW and D = C in case of spatial attention
        coef = torch.bmm(x.transpose(1, 2), bases)
        # column wise softmax ignore batch dim, i.e, on HW dim
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_setp(x, bases, coef)
        return bases, coef

    @torch.no_grad()
    def online_update(self, bases):
        update = bases.mean(dim=0)
        self.bases += self.eta * (update - self.bases)
        # column wise normalization i.e. HW dim
        self.bases = F.normalize(self.bases, dim=1)
        return None

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape

        if self.spatial:
            # spatial attention k
            D = C // self.S  # reduce channels
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R)
            self.register_buffer('bases', bases)
        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)
        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)

        bases = bases.view(B, self.S, D, self.R)

        if not self.rand_init and not self.training and not return_bases:
            self.online_update(bases)

        return x






class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, config, device):
        super().__init__(config)
        self.device = device
        self.inv_t = 1

    def _build_bases(self, B, S, D, R):
        if self.device != "cpuxx": bases = torch.rand((B * S, D, R), device=self.device)
        else:
            bases = torch.cuda.FloatTensor((B * S, D, R))
        bases = F.normalize(bases, dim=1)  # column wise normalization i.e HW dim

        return bases

    @torch.no_grad()
    def local_setp(self, x, bases, coef):
        '''
        Algorithm 2 in paper
        NMF with multiliplicative update.
        '''
        # coef (C/codes)update
        # (B*S, D, N)T @ (B*S, D, R) -> (B*S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)  # D^T @ X
        # (BS, N, R) @ [(BS, D, R)T @ (BS, D, R)] -> (BS, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))  # D^T @ D @ C
        # Multiplicative update
        coef = coef * (numerator / (denominator + 1e-7))  # updated C
        # bases (D/dict) update
        # (BS, D, N) @ (BS, N, R) -> (BS, D, R)
        numerator = torch.bmm(x, coef)  # X @ C^T
        # (BS, D, R) @ [(BS, D, R)T @ (BS, D, R)] -> (BS, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))  # D @ D @ C^T
        # Multiplicative update
        bases = bases * (numerator / (denominator + 1e-7))  # updated D
        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B*S, D, N)T @ (B*S, D, R) -> (B*S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)  # D^T @ X
        # (BS, N, R) @ [(BS, D, R)T @ (BS, D, R)] -> (BS, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))  # D^T @ D @ C
        # Multiplicative update
        coef = coef * (numerator / (denominator + 1e-7))
        return coef


class HamBurger(nn.Module):
    def __init__(self, inChannels, config, device):
        super().__init__()
        self.put_cheese = config['put_cheese']
        C = config["MD_D"]

        # add Relu at end as NMF works of non-negative only
        self.lower_bread = nn.Sequential(nn.Conv2d(inChannels, C, 1),
                                         nn.ReLU(inplace=True)
                                         )
        self.ham = NMF2D(config, device)
        self.cheese = ConvBNRelu(C, C)
        self.upper_bread = nn.Conv2d(C, inChannels, 1, bias=False)

    #     self.init_weights()

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             fan_out //= m.groups
    #             nn.init.normal_(m.weight, std=math.sqrt(2.0/fan_out), mean=0)

    def forward(self, x):
        skip = x.clone()

        x = self.lower_bread(x)
        x = self.ham(x)

        if self.put_cheese:
            x = self.cheese(x)

        x = self.upper_bread(x)
        x = F.relu(x + skip, inplace=True)

        return x

    def online_update(self, bases):
        if hasattr(self.ham, 'online_update'):
            self.ham.online_update(bases)
























class StemConv(nn.Module):
    '''following ConvNext paper'''

    def __init__(self, in_channels, out_channels, bn_momentum=0.99):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            NormLayer(out_channels // 2, norm_type=config['norm_typ']),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            NormLayer(out_channels, norm_type=config['norm_typ'])
        )

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.size()
        # x = x.flatten(2).transpose(1,2) # B*C*H*W -> B*C*HW -> B*HW*C
        return x, H, W


class FFN(nn.Module):
    '''following ConvNext paper'''

    def __init__(self, in_channels, out_channels, hid_channels):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hid_channels, 1)
        self.dwconv = DWConv3x3(hid_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hid_channels, out_channels, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


class BlockFFN(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels, ls_init_val=1e-2, drop_path=0.):
        super().__init__()
        self.norm = NormLayer(in_channels, norm_type=config['norm_typ'])
        self.ffn = FFN(in_channels, out_channels, hid_channels)
        self.layer_scale = LayerScale(in_channels, init_value=ls_init_val)
        self.drop_path = StochasticDepth(p=drop_path)

    def forward(self, x):
        skip = x.clone()

        x = self.norm(x)
        x = self.ffn(x)
        x = self.layer_scale(x)
        x = self.drop_path(x)

        op = skip + x
        return op


class MSCA(nn.Module):

    def __init__(self, dim):
        super(MSCA, self).__init__()
        # input
        self.conv55 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # split into multipats of multiscale attention
        self.conv17_0 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv17_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv111_0 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv111_1 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv211_0 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv211_1 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv11 = nn.Conv2d(dim, dim, 1)  # channel mixer

    def forward(self, x):
        skip = x.clone()

        c55 = self.conv55(x)
        c17 = self.conv17_0(x)
        c17 = self.conv17_1(c17)
        c111 = self.conv111_0(x)
        c111 = self.conv111_1(c111)
        c211 = self.conv211_0(x)
        c211 = self.conv211_1(c211)

        add = c55 + c17 + c111 + c211

        mixer = self.conv11(add)

        op = mixer * skip

        return op


class BlockMSCA(nn.Module):
    def __init__(self, dim, ls_init_val=1e-2, drop_path=0.0):
        super().__init__()

        self.norm = NormLayer(dim, norm_type=config['norm_typ'])
        self.proj1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.msca = MSCA(dim)
        self.proj2 = nn.Conv2d(dim, dim, 1)
        self.layer_scale = LayerScale(dim, init_value=ls_init_val)
        self.drop_path = StochasticDepth(p=drop_path)
        # print(f'BlockMSCA {drop_path}')

    def forward(self, x):
        skip = x.clone()

        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.msca(x)
        x = self.proj2(x)
        x = self.layer_scale(x)
        x = self.drop_path(x)

        out = x + skip

        return out


class StageMSCA(nn.Module):
    def __init__(self, dim, ffn_ratio=4., ls_init_val=1e-2, drop_path=0.0):
        super().__init__()
        # print(f'StageMSCA {drop_path}')
        self.msca_block = BlockMSCA(dim, ls_init_val, drop_path)

        ffn_hid_dim = int(dim * ffn_ratio)
        self.ffn_block = BlockFFN(in_channels=dim, out_channels=dim,
                                  hid_channels=ffn_hid_dim, ls_init_val=ls_init_val,
                                  drop_path=drop_path)

    def forward(self, x):  # input coming form Stem
        # B, N, C = x.shape
        # x = x.permute()
        x = self.msca_block(x)
        x = self.ffn_block(x)

        return x


class MSCANet(nn.Module):
    def __init__(self, in_channnels=3, embed_dims=[32, 64, 460, 256],
                 ffn_ratios=[4, 4, 4, 4], depths=[3, 3, 5, 2], num_stages=4,
                 ls_init_val=1e-2, drop_path=0.0):
        super(MSCANet, self).__init__()
        # print(f'MSCANet {drop_path}')
        self.depths = depths
        self.num_stages = num_stages
        # stochastic depth decay rule (similar to linear decay) / just like matplot linspace
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        cur = 0

        for i in range(num_stages):
            if i == 0:
                input_embed = StemConv(in_channnels, embed_dims[0])
            else:
                input_embed = DownSample(in_channels=embed_dims[i - 1], embed_dim=embed_dims[i])

            stage = nn.ModuleList([StageMSCA(dim=embed_dims[i], ffn_ratio=ffn_ratios[i],
                                             ls_init_val=ls_init_val, drop_path=dpr[cur + j])
                                   for j in range(depths[i])])

            norm_layer = NormLayer(embed_dims[i], norm_type=config['norm_typ'])
            cur += depths[i]

            setattr(self, f'input_embed{i + 1}', input_embed)
            setattr(self, f'stage{i + 1}', stage)
            setattr(self, f'norm_layer{i + 1}', norm_layer)

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            input_embed = getattr(self, f'input_embed{i + 1}')
            stage = getattr(self, f'stage{i + 1}')
            norm_layer = getattr(self, f'norm_layer{i + 1}')

            x, H, W = input_embed(x)

            for stg in stage:
                x = stg(x)

            x = norm_layer(x)
            outs.append(x)

        return outs


__all__ = ['FutureResult', 'SlavePipe', 'SyncMaster']


class FutureResult(object):
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, 'Previous result has\'t been fetched.'
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()

            res = self._result
            self._result = None
            return res






_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])
_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier', 'queue', 'result'])


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret



class SyncMaster(object):
    """An abstract `SyncMaster` object.
    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """
        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def register_slave(self, identifier):
        """
        Register an slave device.
        Args:
            identifier: an identifier, usually is the device id.
        Returns: a `SlavePipe` object which can be used to communicate with the master device.
        """
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).
        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.
        Returns: the message to be sent back to the master device.
        """
        self._activated = True

        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())

        results = self._master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belongs to the master.'

        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)

        for i in range(self.nr_slaves):
            assert self._queue.get() is True

        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)






















def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dementions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])
_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


class _SynchronizedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=3e-4, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)
        self._sync_master = SyncMaster(self._data_parallel_master)

        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

        # customed batch norm statistics
        self.momentum = momentum

    def forward(self, input, weight=None, bias=None):
        # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.
        if not (self._is_parallel and self.training):
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)

        # Resize the input to (B, C, -1).
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)

        # Compute the sum and square-sum.
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)

        # Reduce-and-broadcast the statistics.
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))

        # Compute the output.
        if self.affine:
            if weight is None or bias is None:
                weight = self.weight
                bias = self.bias

            # MJY:: Fuse the multiplication for speed.
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * weight) + _unsqueeze_ft(bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

        # Reshape it.
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id

        # parallel_id == 0 means master device.
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())

        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.get_device() for i in intermediates]

        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)

        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)

        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)

        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i*2:i*2+2])))

        return outputs

    def _add_weighted(self, dest, delta, alpha=1, beta=1, bias=0):
        """return *dest* by `dest := dest*alpha + delta*beta + bias`"""
        return dest * alpha + delta * beta + bias

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data

        return mean, (bias_var + self.eps) ** -0.5



class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):


    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)
























norm_layer = partial(SynchronizedBatchNorm2d, momentum=float(config['SyncBN_MOM']))


class myLayerNorm(nn.Module):
    def __init__(self, inChannels):
        super().__init__()
        self.norm == nn.LayerNorm(inChannels, eps=1e-5)

    def forward(self, x):
        # reshaping only to apply Layer Normalization layer
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B*C*H*W -> B*C*HW -> B*HW*C
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # B*HW*C -> B*H*W*C -> B*C*H*W

        return x


class NormLayer(nn.Module):
    def __init__(self, inChannels, norm_type=config['norm_typ']):
        super().__init__()
        self.inChannels = inChannels
        self.norm_type = norm_type
        if norm_type == 'batch_norm':
            # print('Adding Batch Norm layer') # for testing
            self.norm = nn.BatchNorm2d(inChannels, eps=1e-5, momentum=float(config['BN_MOM']))
        elif norm_type == 'sync_bn':
            # print('Adding Sync-Batch Norm layer') # for testing
            self.norm = norm_layer(inChannels)
        elif norm_type == 'layer_norm':
            # print('Adding Layer Norm layer') # for testing
            self.norm == nn.myLayerNorm(inChannels)
        else:
            raise NotImplementedError

    def forward(self, x):

        x = self.norm(x)

        return x

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.inChannels}, norm_type={self.norm_type})'


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class LayerScale(nn.Module):
    '''
    Layer scale module.
    References:
      - https://arxiv.org/abs/2103.17239
    '''

    def __init__(self, inChannels, init_value=1e-2):
        super().__init__()
        self.inChannels = inChannels
        self.init_value = init_value
        self.layer_scale = nn.Parameter(init_value * torch.ones((inChannels)), requires_grad=True)

    def forward(self, x):
        if self.init_value == 0.0:
            return x
        else:
            scale = self.layer_scale.unsqueeze(-1).unsqueeze(-1)  # C, -> C,1,1
            return scale * x

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.inChannels}, init_value={self.init_value})'


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def stochastic_depth(input: torch.Tensor, p: float,
                     mode: str, training: bool = True):
    if not training or p == 0.0:
        # print(f'not adding stochastic depth of: {p}')
        return input

    survival_rate = 1.0 - p
    if mode == 'row':
        shape = [input.shape[0]] + [1] * (input.ndim - 1)  # just converts BXCXHXW -> [B,1,1,1] list
    elif mode == 'batch':
        shape = [1] * input.ndim

    noise = torch.empty(shape, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    # print(f'added sDepth of: {p}')
    return input * noise


class StochasticDepth(nn.Module):
    '''
    Stochastic Depth module.
    It performs ROW-wise dropping rather than sample-wise.
    mode (str): ``"batch"`` or ``"row"``.
                ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                randomly selected rows from the batch.
    References:
      - https://pytorch.org/vision/stable/_modules/torchvision/ops/stochastic_depth.html#stochastic_depth
    '''

    def __init__(self, p=0.5, mode='row'):
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input):
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self):
        s = f"{self.__class__.__name__}(p={self.p})"
        return s


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def resize(input,
           size=None,
           scale_factor=None,
           mode='bilinear',
           align_corners=None,
           warning=True):
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class DownSample(nn.Module):
    def __init__(self, kernelSize=3, stride=2, in_channels=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(kernelSize, kernelSize),
                              stride=stride, padding=(kernelSize // 2, kernelSize // 2))
        # stride 4 => 4x down sample
        # stride 2 => 2x down sample

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.size()
        # x = x.flatten(2).transpose(1,2)
        return x, H, W


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class DWConv3x3(nn.Module):
    '''Depth wise conv'''

    def __init__(self, dim=768):
        super(DWConv3x3, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class ConvBNRelu(nn.Module):

    @classmethod
    def _same_paddings(cls, kernel):
        if kernel == 1:
            return 0
        elif kernel == 3:
            return 1

    def __init__(self, inChannels, outChannels, kernel=3, stride=1, padding='same',
                 dilation=1, groups=1):
        super().__init__()

        if padding == 'same':
            padding = self._same_paddings(kernel)

        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=kernel,
                              padding=padding, stride=stride, dilation=dilation,
                              groups=groups, bias=False)
        self.norm = NormLayer(outChannels, norm_type=config['norm_typ'])
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x


class SeprableConv2d(nn.Module):
    def __init__(self, inChannels, outChannels, kernal_size=3, bias=False):
        self.dwconv = nn.Conv2d(inChannels, inChannels, kernal_size=kernal_size,
                                groups=inChannels, bias=bias)
        self.pwconv = nn.Conv2d(inChannels, inChannels, kernal_size=1, bias=bias)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)

        return x


class ConvRelu(nn.Module):
    def __init__(self, inChannels, outChannels, kernel=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=kernel, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)

        return x
































class HamDecoder(nn.Module):
    '''SegNext'''

    def __init__(self, outChannels, config, device, enc_embed_dims=[32, 64, 460, 256]):
        super().__init__()

        ham_channels = config['ham_channels']

        self.squeeze = ConvRelu(sum(enc_embed_dims[1:]), ham_channels)
        self.ham_attn = HamBurger(ham_channels, config, device)
        self.align = ConvRelu(ham_channels, outChannels)

    def forward(self, features):
        features = features[1:]  # drop stage 1 features b/c low level
        features = [resize(feature, size=features[-3].shape[2:], mode='bilinear') for feature in features]
        x = torch.cat(features, dim=1)

        x = self.squeeze(x)
        x = self.ham_attn(x)
        x = self.align(x)

        return x


























class SegNext(nn.Module):
    def __init__(self, device, num_classes=1, in_channnels=1, embed_dims=[32, 64, 460, 256],
                 ffn_ratios=[4, 4, 4, 4], depths=[3, 3, 5, 2], num_stages=4,
                 dec_outChannels=256, config=config, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.name = "torchSegNext_"
        self.cls_conv = nn.Sequential(nn.Dropout2d(p=0.1),
                                      nn.Conv2d(dec_outChannels, num_classes, kernel_size=1))
        self.encoder = MSCANet(in_channnels=in_channnels, embed_dims=embed_dims,
                               ffn_ratios=ffn_ratios, depths=depths, num_stages=num_stages,
                               drop_path=drop_path)
        self.decoder = HamDecoder(
            outChannels=dec_outChannels, config=config, device=device, enc_embed_dims=embed_dims)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1.0)
                nn.init.constant_(m.bias, val=0.0)
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, std=math.sqrt(2.0 / fan_out), mean=0)

    def forward(self, x):

        enc_feats = self.encoder(x)
        dec_out = self.decoder(enc_feats)
        output = self.cls_conv(dec_out)  # here output will be B x C x H/8 x W/8
        output = F.interpolate(output, size=x.size()[-2:], mode='bilinear', align_corners=True)  # now its same as input
        #  bilinear interpol was used originally
        return output