import torch.nn as nn
import torch


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # B x inplanes x H x W

        out = self.conv1(x)  # B x planes x H1 x W1 (Kernel size 3, Padding 1: (x-1)//stride+1)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # B x planes x H2 x W2 (Kernel size 3, Stride length 1, Padding 1: x)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=64,
                 dropout=0.0, inplanes=64, global_pool=True):
        super(ResNet, self).__init__()
        self.initial_pool = False
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.outplanes = 512  # Set to output of layer4, which is inplanes * 8

        # handle classifier creation
        if num_classes is not None:
            self.cls_fn = nn.Linear(self.outplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        # Inplanes to planes for first, planes to planes * block expansion for downsample,
        # planes * block expansion to planes for the rest
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, kd=False):
        embed = self.embed(x)  # B x inplanes * 8
        embed = self.dropout(embed)
        x = embed
        x = self.cls_fn(x)  # B x num_classes
        if kd:
            return x, embed
        else:
            return x

    def embed(self, x, param_dict=None):  # B x 3 x H x W
        x = self.conv1(x)  # B x inplanes x H0 x W0 (Kernel size 5, Stride length 2, Padding 1: (x-3)//2+1)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)
        x = self.layer1(x)  # B x inplanes x H1 x W1 (Kernel size 3, Stride length 1, Padding 1: x)
        x = self.layer2(x)  # B x inplanes * 2 x H2 x W2 (Kernel size 3, Stride length 2, Padding 1: (x-1)//2+1)
        x = self.layer3(x)  # B x inplanes * 4 x H3 x W3 (Kernel size 3, Stride length 2, Padding 1: (x-1)//2+1)
        x = self.layer4(x)  # B x inplanes * 8 x H4 x W4 (Kernel size 3, Stride length 2, Padding 1: (x-1)//2+1)

        x = self.avgpool(x)  # B x inplanes * 8 x 1 x 1
        return x.squeeze()  # B x inplanes * 8

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.named_parameters()]

def resnet18(n_cls, nf=64):
    """
        Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=n_cls, inplanes=nf)
    return model


def load_model(model, save_path):
    device = model.get_parameters()[0].device
    ckpt_dict = torch.load(save_path, map_location=device)['state_dict']
    shared_state = {k: v for k, v in ckpt_dict.items() if 'cls' not in k}  # All except classifier
    model.load_state_dict(shared_state, strict=False)
    print('Loaded shared weights from {}'.format(save_path))
    return model


def change_param_prefix(params, old_prefix, new_prefix):
    _len = len(old_prefix)
    names = list(params.keys())
    for name in names:
        if name.startswith(old_prefix):
            new_name = f"{new_prefix}{name[_len:]}"
            params[new_name] = params[name]
            del params[name]

    return params


def load_model2(model, save_path):
    device = model.get_parameters()[0].device
    ckpt_dict = torch.load(save_path, map_location=device)['model']
    ckpt_dict = change_param_prefix(ckpt_dict, "backbone.", "")
    shared_state = {k: v for k, v in ckpt_dict.items() if 'cls' not in k and k not in ["layer.bias", "layer.weight"]}
    model.load_state_dict(shared_state, strict=False)
    print('Loaded shared weights from {}'.format(save_path))
    return model
