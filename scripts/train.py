import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append('..')

import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
from torch.utils.data import DataLoader
import random
from torch.optim import lr_scheduler

from network.model_vit import BirdViT
from data.dataset import BirdCLEFDataset
from network.make_model import AudioModel
from utils.utils import Logger, AverageMeter, calculate_metrics
from sklearn.metrics import f1_score


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_model(model, criterion, optimizer, epoch):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    training_process = tqdm(train_loader)
    for i, (XI, label) in enumerate(training_process):
        if i > 0:
            training_process.set_description(
                "Train Epoch: %d, Loss: %.4f, Acc: %.4f" % (epoch, losses.avg.item(), accuracies.avg.item()))

        x = Variable(XI.cuda(device_id))
        label = Variable(label.cuda(device_id))
        # label = Variable(torch.LongTensor(label).cuda(device_id))
        # Forward pass: Compute predicted y by passing x to the model
        y_out = model(x)
        # Compute and print loss
        loss = criterion(y_out, label)
        # print(nn.Softmax(dim=1)(y_pred), y_pred)
        acc = calculate_metrics(nn.Softmax(dim=1)(y_out).cpu(), label.cpu())
        losses.update(loss.cpu(), x.size(0))
        accuracies.update(acc, x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    train_logger.log(phase="train", values={
        'epoch': epoch,
        'loss': format(losses.avg.item(), '.4f'),
        'acc': format(accuracies.avg.item(), '.4f'),
        'lr': optimizer.param_groups[0]['lr'],
        'f1': 0.0
    })
    print("Train:\t Loss:{0:.4f} \t Acc:{1:.4f} \t lr:{2:.4f}".
          format(losses.avg, accuracies.avg, optimizer.param_groups[0]['lr']))


def eval_model(model, epoch, eval_loader, is_save=True):
    eval_criterion = nn.CrossEntropyLoss()
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    eval_process = tqdm(eval_loader)

    y_preds = []
    labels = []
    with torch.no_grad():
        for i, (img, label) in enumerate(eval_process):
            if i > 0:
                eval_process.set_description("Val Epoch: %d, Loss: %.4f, Acc: %.4f" %
                                             (epoch, losses.avg.item(), accuracies.avg.item()))
            img, label = Variable(img.cuda(device_id)), Variable(label.cuda(device_id))
            y_out = model(img)

            loss = eval_criterion(y_out, label)
            acc = calculate_metrics(nn.Softmax(dim=1)(y_out).cpu(), label.cpu())

            _, y_pred = torch.max(y_out, 1)
            y_preds.extend(y_pred.view(-1).cpu().detach().numpy())
            labels.extend(label.view(-1).cpu().detach().numpy())

            losses.update(loss.cpu(), img.size(0))
            accuracies.update(acc, img.size(0))

    f1 = f1_score(labels, y_preds, average='macro')

    if is_save:
        train_logger.log(phase="val", values={
            'epoch': epoch,
            'loss': format(losses.avg.item(), '.4f'),
            'acc': format(accuracies.avg.item(), '.4f'),
            'lr': optimizer.param_groups[0]['lr'],
            'f1': f1
        })
    print("Val:\t Loss:{0:.4f} \t Acc:{1:.4f} \t F1:{2:.4f} \t lr:{3:.4f}".
          format(losses.avg, accuracies.avg, f1, optimizer.param_groups[0]['lr']))
    return f1  # accuracies.avg


parser = argparse.ArgumentParser(description="CSIG audio  @cby Training")
parser.add_argument("--device_id", default=0, help="Setting the GPU id", type=int)
parser.add_argument("--k", default=0, help="The value of K Fold", type=int)
parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()

if __name__ == "__main__":
    # 设置随机数种子
    setup_seed(20)
    # set dataset
    train_path = '/data1/chenby/dataset/BirdClef2022'
    k = args.k
    batch_size = 256
    test_batch_size = batch_size
    epoch_start = 1
    num_epochs = epoch_start + 100
    device_id = args.device_id
    lr = 1e-3
    num_workers = 8

    model_name = 'efficientnet-b0'  # efficientnet-b0
    # model_name = 'vit'  # efficientnet-b0
    writeFile = '../output/logs/' + model_name + f'_k{k}'
    store_name = '../output/weights/' + model_name + f'_k{k}'

    audio_model = AudioModel(model_name=model_name)
    # audio_model = BirdViT(depth=24, heads=4, dim=64, mlp_dim=256, seq_len=313, emb_dim=224)
    model_path = None
    # model_path = '/pubdata/chenby/weights/vgg_16_v4/audio9_acc0.9988.pth'
    if model_path is not None:
        audio_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print('Model found in {}'.format(model_path))
    else:
        print('No network found, initializing random network.')
    audio_model = audio_model.cuda(device_id)

    criterion = nn.CrossEntropyLoss()
    is_training = True
    if is_training:
        if store_name and not os.path.exists(store_name):
            os.makedirs(store_name)
        train_dataset = BirdCLEFDataset(root_path=train_path, fold=k, phase='train')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                  num_workers=num_workers)
        eval_dataset = BirdCLEFDataset(root_path=train_path, fold=k, phase='val')
        eval_loader = DataLoader(eval_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

        # optimizer = optim.SGD(audio_model.parameters(), lr=lr, momentum=0.9)  # 原始使用
        optimizer = optim.AdamW(audio_model.parameters(), lr=lr, weight_decay=4e-5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6, last_epoch=-1)

        train_logger = Logger(model_name=writeFile, header=['epoch', 'loss', 'acc', 'lr', 'f1'])
        best_f1 = 0.0 if epoch_start == 1 else eval_model(audio_model, epoch_start - 1, eval_loader, is_save=False)
        for epoch in range(epoch_start, num_epochs):
            train_model(audio_model, criterion, optimizer, epoch)
            f1 = eval_model(audio_model, epoch, eval_loader)
            if best_f1 < f1:
                best_f1 = f1
                torch.save(audio_model.state_dict(), '{}/{}_f1_{:.4f}.pth'.format(store_name, 'audio' + str(epoch), f1))
            print("Current Best F1:{0:.4f}".format(best_f1))
    else:
        eval_dataset = BirdCLEFDataset(root_path=train_path, fold=k, phase='val')
        eval_loader = DataLoader(eval_dataset, batch_size=test_batch_size, shuffle=False)
        acc = eval_model(audio_model, -1, eval_loader, is_save=False)
