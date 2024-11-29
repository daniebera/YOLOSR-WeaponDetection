import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class DeepSuperRes(nn.Module):
    def __init__(self, ch, c1=128, c2=512, factor=2, srdepth='deep', sync_bn=True):
        super(DeepSuperRes, self).__init__()
        assert srdepth == 'deep' or srdepth == 'early'
        self.srdepth = 1 if srdepth == 'deep' else 0

        # Fixme: removed synchBN as yolo should auto convert the whole model on need
        # if sync_bn == True:
        #     BatchNorm = SynchronizedBatchNorm2d
        # else:
        #     BatchNorm = nn.BatchNorm2d
        BatchNorm = nn.BatchNorm2d

        self.sr_encoder = Encoder(c1, c2, self.srdepth)
        self.edsr = EDSR(num_channels=ch, input_channel=64, factor=16 if self.srdepth else 8)  # Fixme: factor = 16 for Solution 1
        self.factor = factor

    def forward(self, low_level_feat, high_level_feat):
        out_sr = self.sr_encoder(high_level_feat, low_level_feat, self.factor)
        out_sr_up = self.edsr(out_sr)

        return out_sr_up


class Encoder(nn.Module):
    def __init__(self, c1, c2, srdepth):
        super(Encoder, self).__init__()

        # Todo: Verify if bias should be = False
        # Fixme: remove conv1,conv2 for Solution2
        self.srdepth = srdepth
        if self.srdepth:
            self.conv1 = nn.Conv2d(c1, c1 // 2, 1, bias=False)
            self.conv2 = nn.Conv2d(c2, c2 // 2, 1, bias=False)
            self.relu = nn.ReLU()

        self.last_conv = nn.Sequential(nn.Conv2d((c1 + c2) // 2, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.ReLU(),
                                       nn.Conv2d(128, 64, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, high_level_feat, low_level_feat, factor):
        # Fixme: remove conv1,conv2 for Solution2
        if self.srdepth:
            low_level_feat = self.conv1(low_level_feat)
            low_level_feat = self.relu(low_level_feat)

            high_level_feat = self.conv2(high_level_feat)
            high_level_feat = self.relu(high_level_feat)

        # Fixme: issue https://discuss.pytorch.org/t/deterministic-behavior-using-bilinear2d/131355
        high_level_feat = F.interpolate(high_level_feat,
                                        size=[i * (factor // 2) for i in low_level_feat.size()[2:]],
                                        # mode='bilinear',
                                        # align_corners=True
                                        )
        if factor > 1:
            low_level_feat = F.interpolate(low_level_feat,
                                           size=[i * (factor // 2) for i in low_level_feat.size()[2:]],
                                           # mode='bilinear',
                                           # align_corners=True
                                           )
        out = torch.cat((high_level_feat, low_level_feat), dim=1)
        out = self.last_conv(out)

        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            # Fixme: removed synchBN as yolo should auto convert the whole model on need
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feat))
                if act:
                    m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if act:
                m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()

        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class EDSR(nn.Module):
    def __init__(self, num_channels=3, input_channel=64, factor=4, width=64, depth=16, kernel_size=3, conv=default_conv):
        super(EDSR, self).__init__()

        n_resblock = depth
        n_feats = width
        kernel_size = kernel_size
        scale = factor
        act = nn.ReLU()

        # define head module
        m_head = [conv(input_channel, n_feats, kernel_size)]

        # define body module
        m_body = [ResBlock(conv,
                           n_feats,
                           kernel_size,
                           act=act,
                           res_scale=1.) for _ in range(n_resblock)]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [Upsampler(conv, scale, n_feats, act=False),
                  conv(n_feats, num_channels, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
