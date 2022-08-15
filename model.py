from turtle import forward
import torch
import torch.nn as nn
from torch.autograd import Function


def Linear(in_features, out_features, activation=None, batch_norm=False, dropout=0.):
    if batch_norm:
        layers  = [ nn.Linear(in_features, out_features, bias=False) ]
        layers += [ nn.BatchNorm1d(out_features) ]
    else:
        layers  = [ nn.Linear(in_features, out_features, bias=True) ]

    if activation is not None:
        layers += [ getattr(nn, activation['name'])(**activation['args']) ]

    if dropout is not None and 0. < dropout < 1.:
        layers += [ nn.Dropout(dropout, inplace=True) ] 

    return nn.Sequential(*layers)


def Conv2d(in_channels, out_channels, kernel_size, padding=0, groups=1, activation=None, batch_norm=False):
    if batch_norm:
        layers  = [ nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, groups=groups, bias=False) ]
        layers += [ nn.BatchNorm2d(out_channels) ]
    else:
        layers  = [ nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, groups=groups, bias=True) ]

    if activation is not None:
        layers += [ getattr(nn, activation['name'])(**activation['args']) ]

    return nn.Sequential(*layers)


def SepConv2d(in_channels, out_channels, kernel_size, padding=0, expand_ratio=1., activation=None, batch_norm=False):
    hidden_channels = max(int(in_channels * expand_ratio), 1)
    
    layers = []
    if hidden_channels != in_channels:
        layers += [ Conv2d(in_channels, hidden_channels, 1, activation=activation, batch_norm=batch_norm) ]

    layers += [ 
            Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, \
                    groups=hidden_channels, activation=activation, batch_norm=batch_norm),
            Conv2d(hidden_channels, out_channels, 1, batch_norm=batch_norm)
            ]

    return nn.Sequential(*layers)


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        output = ctx.alpha * grad.neg()

        return output, None


class CNN(nn.Module):
    def __init__(self, input_size, channels, n_class, n_domain, separable=False, expand_ratio=1., \
            activation=None, batch_norm=False, alpha=1.):
        super(CNN, self).__init__()

        _sample = torch.randn(2, input_size[0], input_size[1])

        self.downpool  = nn.AvgPool1d(2)

        _sample = self.downpool(_sample)

        conv = []
        in_channels = 1
        for out_channels in channels:
            if separable:
                conv += [ SepConv2d(in_channels, out_channels, 3, padding=1, \
                        expand_ratio=expand_ratio, activation=activation, batch_norm=batch_norm) ]
            else:
                conv += [ Conv2d(in_channels, out_channels, 3, padding=1, \
                        activation=activation, batch_norm=batch_norm) ]
            conv += [ nn.AvgPool2d(2) ]
            in_channels = out_channels
        self.conv = nn.Sequential(*conv)

        _sample = _sample.unsqueeze(1)
        _sample = self.conv(_sample)
        _batch, _n_channels, _n_features, _n_frames = _sample.size()

        in_features = _n_channels * _n_features * _n_frames
        self.cls_linear = Linear(in_features, n_class)
        self.dmn_linear = Linear(in_features, n_domain)

        self.alpha = alpha

    def forward(self, x):
        x = self.downpool(x)

        x = x.unsqueeze(1)
        hidden = self.conv(x)

        hidden = hidden.flatten(start_dim=1)
        cls_output = self.cls_linear(hidden)

        rev_hidden = GradReverse.apply(hidden, self.alpha)
        dmn_output = self.dmn_linear(rev_hidden)

        return cls_output, dmn_output

    def num_params(self): 
        return sum([ param.numel() for name, param in self.named_parameters() if 'dmn_linear' not in name ])



if __name__ == '__main__':
    m1 = CNN(
        input_size=[251, 127],
        channels  =[16, 16, 16, 16, 16],
        n_class   =3,
        n_domain  =2,
        batch_norm=True,
        activation={'name': 'LeakyReLU', 'args': {'negative_slope': 0.2, 'inplace':True}}
    )
    cnt1 = m1.num_params()
    print(cnt1)

    m2 = CNN(
        input_size  =[251, 127],
        channels    =[16, 16, 16, 16, 16],
        n_class     =3,
        n_domain    =2,
        batch_norm  =True,
        activation  ={'name': 'LeakyReLU', 'args': {'negative_slope': 0.2, 'inplace':True}},
        separable   =True,
        expand_ratio=1.
    )
    cnt2 = m2.num_params()
    print(cnt2)
    
    print(f'Compression Ratio: {(cnt1 - cnt2) / cnt1 * 100:4.2f}')