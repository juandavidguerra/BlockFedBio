import pandas as pd
import csv
import os
import sys
import torch
import shutil
import pickle
import torch
import random
import numpy as np
import torch.nn.functional as F 
from torchvision import datasets, transforms
from torch.utils.data import Dataset 

def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.

    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))


def get_id_label_map(meta_file):
    N_IDENTITY = 9131  # total number of identities in VGG Face2
    N_IDENTITY_PRETRAIN = 8631  # the number of identities used in training by Caffe
    identity_list = meta_file
    df = pd.read_csv(identity_list, sep=',', encoding="utf-8", engine ='python',skiprows=1)
    df["class"] = -1
    df.loc[df["Flag"] == 1, "class"] = range(N_IDENTITY)
    df.loc[df["Flag"] == 0, "class"] = range(N_IDENTITY-N_IDENTITY_PRETRAIN)
    # print(df)
    key = df["Class_ID"].values
    val = df["class"].values
    id_label_dict = dict(zip(key, val))
    return id_label_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    output_sorted, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def client_update(client_model, optimizer, train_loader, epoch):
    """
    This function updates/trains client model on client data
    """
    client_model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    return loss.item()

def server_aggregate(global_model, client_models):
    """
    This function has aggregation method 'mean'
    """
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

def test(global_model, test_loader):
    """
    This function test the global model on test data and returns test loss and test accuracy
    """
    global_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = global_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    return test_loss, acc

def get_cifar10():
    data_train = datasets.CIFAR10('./data',  train = True, download = True)
    data_test = datasets.CIFAR10('./data',  train = False, download = True)
    x_train, y_train = data_train.data.transpose((0,3,1,2)), np.array(data_train.targets)
    x_test, y_test = data_test.data.transpose((0,3,1,2)), np.array(data_test.targets)
    return x_train, y_train, x_test, y_test
    
def get_celeba():
    return

def get_vggface():
    return

def get_defult_data_transforms(train = True, verbose = True):
    transforms_train = {
        'cifar10' : transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32,padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])}
    transforms_eval = {
        'cifar10' : transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()])}
    if verbose:
        print('\n Data prepocesing')
        for transformation in transforms_train['cifar10'].transforms:
            print('-', transformation)
        print()
    return (transforms_train['cifar10'],transforms_eval['cifar10'])

def shuffle_list(data):
    for i in range(len(data)):
        tmp_len= len(data[i][0])
        index = [i for i in range(tmp_len)]
        random.shuffle(index)
        data[i][0],data[i][1] = shuffle_list_data(data[i][0],data[i][1])
    return data

def shuffle_list_data(x, y):
    inds = list(range(len(x)))
    random.shuffle(inds)
    return x[inds], y[inds]

def baseline_data(num):
    '''
    Create return baseline data loader for
    '''
    xtrain, ytrain, xtmp, ytmp = get_cifar10()
    x, y = shuffle_list_data(xtrain, ytrain)
    x, y = x[:num], y[:num]
    transform, _ = get_defult_data_transforms(train=True, verbose=False)
    loader = torch.utils.data.DataLoader(CustomImageDataset(x,y,transform), batch_size=16, shuffle=True )
    return loader

class CustomImageDataset(Dataset):
    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]
        if self.transforms is not None:
            img = self.transforms(img)
        return (img, label)

    def __len__(self):
        return self.inputs.shape[0]

