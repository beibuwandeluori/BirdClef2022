import torch
import torch.nn as nn
import torchvision
from efficientnet_pytorch import EfficientNet
import timm
import torch.nn.functional as F


def init_weights(module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.BatchNorm1d)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02)


def get_efficientnet(model_name='efficientnet-b0', num_classes=152):
    net = EfficientNet.from_pretrained(model_name)
    # net = EfficientNet.from_name(model_name)
    in_features = net._fc.in_features
    net._fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    return net


def get_efficientnet_ns(model_name='tf_efficientnet_b0_ns', pretrained=True, num_classes=152):
    """
     # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    :param model_name:
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    n_features = net.classifier.in_features
    net.classifier = nn.Linear(n_features, num_classes)

    return net


def get_resnet(modelchoice='resnet50', num_classes=2):
    if modelchoice == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif modelchoice == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif modelchoice == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif modelchoice == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
    elif modelchoice == 'resnext101_32x8d':
        model = torchvision.models.resnext101_32x8d(pretrained=True)
    elif modelchoice == 'resnext50_32x4d':
        model = torchvision.models.resnext50_32x4d(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    return model


def get_vgg(modelchoice='vgg16', num_classes=2):
    if modelchoice == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
    elif modelchoice == 'vgg19':
        model = torchvision.models.vgg19(pretrained=True)
    # print(model)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    return model


class AudioModel(nn.Module):

    def __init__(self, model_name='efficientnet-b0', num_classes=512):
        super(AudioModel, self).__init__()
        self.model_name = model_name

        if 'ns' in self.model_name:
            self.model_name = model_name
            self.model = get_efficientnet_ns(model_name=model_name, num_classes=num_classes)
            self.model.conv_stem.in_channels = 1
            self.model.conv_stem.weight = torch.nn.Parameter(self.model.conv_stem.weight[:, 0:1:, :, :])
        elif 'efficientnet' in model_name:
            self.model_name = model_name
            self.model = get_efficientnet(model_name=model_name, num_classes=num_classes)
            self.model._conv_stem.in_channels = 1
            self.model._conv_stem.weight = torch.nn.Parameter(self.model._conv_stem.weight[:, 0:1:, :, :])
        elif 'res' in self.model_name:
            self.model = get_resnet(modelchoice=model_name, num_classes=num_classes)
            self.model.conv1.in_channels = 1
            self.model.conv1.weight = torch.nn.Parameter(self.model.conv1.weight[:, 0:1:, :, :])
        elif 'vgg' in self.model_name:
            self.model = get_vgg(modelchoice=model_name, num_classes=num_classes)
            self.model.features[0].in_channels = 1
            self.model.features[0].weight = torch.nn.Parameter(self.model.features[0].weight[:, 0:1:, :, :])

        print(f'Model name is {self.model_name}!')

    def forward(self, x):
        x = self.model(x)

        return x


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + \
               '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
               ', ' + 'eps=' + str(self.eps) + ')'


class BirdCLEFModel(nn.Module):
    def __init__(self, model_name, embedding_size=1024, pretrained=True, num_classes=152):
        super(BirdCLEFModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        if 'ns' in self.model_name:
            self.model_name = model_name
            self.model = get_efficientnet_ns(model_name=model_name, num_classes=num_classes)
            self.model.conv_stem.in_channels = 1
            self.model.conv_stem.weight = torch.nn.Parameter(self.model.conv_stem.weight[:, 0:1:, :, :])
        elif 'efficientnet' in model_name:
            self.model_name = model_name
            self.model = get_efficientnet(model_name=model_name, num_classes=num_classes)
            self.model._conv_stem.in_channels = 1
            self.model._conv_stem.weight = torch.nn.Parameter(self.model._conv_stem.weight[:, 0:1:, :, :])

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.embedding = nn.Linear(in_features, embedding_size)
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features)
        output = self.fc(embedding)
        return output


if __name__ == '__main__':
    network = AudioModel(model_name='efficientnet-b2', num_classes=152)  # efficientnet-b0 resnet50
    network = network.to(torch.device('cpu'))
    print(network)

    from torchsummary import summary
    input_s = (1, 500, 96)
    print(summary(network, input_s, device='cpu'))
