### libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import numpy as np
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./runs/')

### configuration
torch.backends.cudnn.benchmark=True
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING']='1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### architecture
model_name = 'mobilenet_v2'# vgg19/resnet50/mobilenet_v2
dataset= 'CelebA-10' # CelebA-10 or VGG-Face2-10

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

### transform
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

### loader
data_dir = dataroot + dataset #VGG-Face2-10
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

### train & test function
def train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    val_acc_history = []
    val_losses = []
    train_losses = []
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]: # Iterate over data
                inputs = inputs.to(device)
                labels = labels.to(device)                
                optimizer.zero_grad() # zero the parameter gradients
                with torch.set_grad_enabled(phase == 'train'): # track history if only in train
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)                    
                    if phase == 'train': # backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()                
                running_loss += loss.item() * inputs.size(0) # statistics                
                running_corrects += torch.sum(preds == labels.data) # statistics
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            writer.add_scalar('Acc', epoch_acc, epoch)
            writer.add_scalar('Loss', epoch_loss, epoch)
            print('Epoch {}/{}, {} Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs, phase, epoch_loss, epoch_acc))            
            if phase == 'val' and epoch_acc > best_acc: # deep copy the model
                best_acc = epoch_acc
                #best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print(model_name, dataset)
    #model.load_state_dict(best_model_wts) # load best model weights
    return model, val_acc_history

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
        model_ft = models.quantization.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "vgg19":
        model_ft = models.vgg19(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    elif model_name == "mobilenet_v2":
        model_ft = models.quantization.mobilenet_v2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs,num_classes)
    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft

### model selection
model_ft = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True)
model_ft = model_ft.to(device)

### loss function other parameters
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr, momentum) # Observe that all parameters are being optimized
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size, gamma) # Decay LR by a factor of 0.1 every 7 epochs

### cnn train

accuracies = []
examples = []

model = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs)



