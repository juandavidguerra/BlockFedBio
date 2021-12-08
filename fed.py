from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

plt.ion() 

##architecture
model_name = 'resnet50'# vgg19 or resnet50
dataset= 'VGG-Face2-10' # CelebA-10 or VGG-Face2-10

#Inputs
batch_size=64
num_workers=8
num_classes=10
num_epochs=2
lr=0.001
momentum=0.9
step_size=7
gamma=0.1

num_clients = 4
num_selected = 2
num_rounds = 2

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),}

data_dir = '~/Documents/datasets/'+ dataset #VGG-Face2-10
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}




traindata_split = torch.utils.data.random_split(image_datasets['train'], [int(len(image_datasets['train']) / num_clients) for _ in range(num_clients)])
train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]

###keu
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}
##


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes










def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):

        for phase in ['train', 'val']:
            if phase == 'train':############
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode#####
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]: # Iterate over data########
                inputs = inputs.to(device)######
                labels = labels.to(device)  ######              
                optimizer.zero_grad() # zero the parameter gradients#########

                with torch.set_grad_enabled(phase == 'train'): # track history if only in train
                    outputs = model(inputs) #####3
                    _, preds = torch.max(outputs, 1) #####
                    loss = criterion(outputs, labels) #######
                    
                    if phase == 'train': # backward + optimize only if in training phase
                        loss.backward() #######
                        optimizer.step() ########
                
                running_loss += loss.item() * inputs.size(0) # statistics #####
                running_corrects += torch.sum(preds == labels.data) # statistics#####
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('Epoch {}/{}, {} Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs, phase, epoch_loss, epoch_acc))            
            if phase == 'val' and epoch_acc > best_acc: # deep copy the model
                best_acc = epoch_acc
                #best_model_wts = copy.deepcopy(model.state_dict())
    return epoch_loss, epoch_acc

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))    
    #model.load_state_dict(best_model_wts) # load best model weights
    return model



def server_aggregate(global_model, client_models):
    """
    This function has aggregation method 'mean'
    """
    ### This will take simple mean of the weights of models ###
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())




def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

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
    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft


model, criterion, optimizer, scheduler, num_epochs


global_model = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True)
global_model = global_model.to(device)
client_models = [ initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True).to(device) for _ in range(num_selected)]
for model in client_models:
    model.load_state_dict(global_model.state_dict())
opt = [optim.SGD(model.parameters(), lr=0.1) for model in client_models]
losses_train = []
losses_test = []
acc_train = []
acc_test = []

for r in range(num_rounds):
    # select random clients
    client_idx = np.random.permutation(num_clients)[:num_selected]
    # client update
    loss = 0
    for i in range(num_selected):
        loss += train_model(client_models[i], opt[i], train_loader[client_idx[i]], num_epochs)
    
    losses_train.append(loss)
    # server aggregate
    server_aggregate(global_model, client_models)
    
    test_loss, acc = train_model(global_model, dataloaders)
    losses_test.append(test_loss)
    acc_test.append(acc)
    print('%d-th round' % r)
    print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / num_selected, test_loss, acc))

"""

model_ft = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr, momentum) # Observe that all parameters are being optimized
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size, gamma) # Decay LR by a factor of 0.1 every 7 epochs
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs)


"""