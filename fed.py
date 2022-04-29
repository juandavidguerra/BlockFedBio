###libraries
from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets, models

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./runs/')

### configuration

torch.backends.cudnn.benchmark=True
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING']='1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### architecture
model_name = 'mobilenet_v2'# vgg19 or resnet50 mobilenet_v2
dataset= 'VGG-Face2-10' # CelebA-10 or VGG-Face2-10

### inputs
batch_size=64
num_workers=8
num_classes=10
num_epochs=100
lr=0.001
momentum=0.9
step_size=7
gamma=0.1
dataroot = '/home/juan/Documents/datasets/'

### fed inputs
num_clients = 4
num_selected = 2
num_rounds = 10

### transform
transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

### loader
data_dir = dataroot + dataset #VGG-Face2-10
traindata= datasets.ImageFolder(os.path.join(data_dir,'train'),transform=transform_train)
testdata= datasets.ImageFolder(os.path.join(data_dir,'val'),transform=transform_test)
traindata_split = torch.utils.data.random_split(traindata, [int(len(traindata)/ num_clients) for _ in range(num_clients)])
train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in traindata_split]
test_loader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=True, num_workers=num_workers)

### train function
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

### test function
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
    writer.add_scalar('Acc',acc, num_epochs)
    writer.add_scalar('Loss', test_loss, num_epochs)
    return test_loss, acc

### agregate function
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

### utils
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

### choose model
def initialize_model(model_name, num_classes, feature_extract, use_pretrained):
    model_ft = None
    print(model_name, dataset)
    if model_name == "resnet50":
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "vgg19":
        model_ft = models.vgg19(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    elif model_name == "mobilenet_v2":
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs,num_classes)
    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft


### model selection
model_ft = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True)
global_model = model_ft.to(device)
client_models = [ model_ft.to(device) for _ in range(num_selected)]
for model in client_models:
    model.load_state_dict(global_model.state_dict()) 

### loss function other parameters
opt = [optim.SGD(model.parameters(), lr=lr) for model in client_models]
losses_train = []
losses_test = []
acc_train = []
acc_test = []

### federated train
for r in range(num_rounds):
    client_idx = np.random.permutation(num_clients)[:num_selected]
    loss = 0
    for i in tqdm(range(num_selected)):
        loss += client_update(client_models[i], opt[i], train_loader[client_idx[i]], epoch=num_epochs)    
    losses_train.append(loss)
    server_aggregate(global_model, client_models)
    test_loss, acc = test(global_model, test_loader)
    losses_test.append(test_loss)
    acc_test.append(acc)
    #writer.add_scalar('Acc Server',acc_test, num_rounds)
    #writer.add_scalar('Loss Server', losses_test, num_rounds)
    print('%d-th round' % r)
    print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / num_selected, test_loss, acc))