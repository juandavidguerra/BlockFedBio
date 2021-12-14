from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

#from model.vgg import VGG 
from utils import client_update, server_aggregate, test
from data.celeba2k import CelebA as Cb

torch.backends.cudnn.benchmark=True
torch.cuda.empty_cache()
CUDA_LAUNCH_BLOCKING=1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = 'celeba'
workers = 8
num_clients = 4
num_selected = 2
num_rounds = 2
epochs = 2
batch_size = 32
lr=0.01
num_classes = 2001
m1 = 512*4*4
m2= 4096

dataroot = '/home/juan/Documents/datasets/'

transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                       transforms.Resize((128, 128)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])

traindata = Cb(root=dataroot, split= 'train', transform=transform )
testdata = Cb(root=dataroot, split= 'valid', transform=transform) 
traindata_split = torch.utils.data.random_split(traindata, [int(len(traindata)/ num_clients) for _ in range(num_clients)])
train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True, num_workers=workers) for x in traindata_split]
test_loader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=True, num_workers=workers)

cfg = {'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],}

class VGG(nn.Module):
    def __init__(self,vgg_name, num_classes, m1, m2):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(m1,m2),
            nn.ReLU(True),
            nn.Linear(m2,m2),
            nn.ReLU(True),
            nn.Linear(m2,num_classes))
        
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        output = F.log_softmax(out, dim=1)
        return output

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

global_model =  VGG('VGG16',num_classes=num_classes,m1=m1,m2=m2).to(device)
client_models = [ VGG('VGG16',num_classes=num_classes,m1=m1,m2=m2).to(device) for _ in range(num_selected)]
for model in client_models:
    model.load_state_dict(global_model.state_dict()) 
opt = [optim.SGD(model.parameters(), lr=lr) for model in client_models]

losses_train = []
losses_test = []
acc_train = []
acc_test = []

# Runnining FL
for r in range(num_rounds):
    client_idx = np.random.permutation(num_clients)[:num_selected]
    loss = 0
    for i in tqdm(range(num_selected)):
        loss += client_update(client_models[i], opt[i], train_loader[client_idx[i]], epoch=epochs)    
    losses_train.append(loss)
    server_aggregate(global_model, client_models)
    test_loss, acc = test(global_model, test_loader)
    losses_test.append(test_loss)
    acc_test.append(acc)
    print('%d-th round' % r)
    print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / num_selected, test_loss, acc))