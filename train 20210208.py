# import libraries from Part 1
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, models, transforms
import numpy as np
import pandas as pd
from collections import OrderedDict
from PIL import Image
import json
import argparse
# ref (https://stackoverflow.com/questions/58891777/nameerror-name-argparse-is-not-defined)
from argparse import ArgumentParser
from collections import OrderedDict

#argparse (https://www.youtube.com/watch?v=cdblJqEUDNo / https://youtu.be/TzhRdfBQ-Xo)
#Options:
# 1. Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# 2. Choose architecture: python train.py data_dir --arch "vgg13"
# 3. Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# 4. Use GPU for training: python train.py data_dir --gpu
# Total 7 arguments: data_dir (namdatory), --save_dir, --arch, --learning_rate, --hidden_units, --epochs, --gpu 

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', type = str, metavar = '', help = 'Directory of files. Required.')
parser.add_argument('--save_dir', type = str, metavar = '', help = 'Save directory. Default is Checkpoint', default = 'Checkpoint')
parser.add_argument('--arch', type = str, metavar = '', help = 'Choose vgg architecture e.g. vgg16, vgg13. Default is vgg16', default = 'vgg16')
parser.add_argument('--lrn', type = float, metavar = '', help = 'Learn rate. Default is 0.002', default = 0.002)
parser.add_argument('--hidden_units', type = int, metavar = '', help = 'Hidden units. Default is 1024', default = 1024)
parser.add_argument('--epochs', type = int, metavar = '', help = 'Number of epochs. Default is 3', default = 3)
parser.add_argument('--GPU', type = str, metavar = '', help = 'Use GPU/cuda. Default is GPU/cuda', default = 'GPU')
parser.add_argument('--batch_size', type = int, metavar = '', help = 'Enter batch size', default = 128)

args = parser.parse_args()
#adding a list of checks for user as recommended on review 2/7/2021
if(args.save_dir == 'help'):
    print("Enter your save directory without the preceding slash '/'")
    print("Example: ")
    print("  Checkpoint      <- This is good")
    print("  /Checkpoint     <- this will not work")
    quit()

if(args.arch == 'help'):
    print("Choose an available CNN network from the following:")                    
    print("-  VGG11")
    print("-  VGG13")
    print("-  VGG16 (default)")
    print("-  VGG19")   
    print("-  densenet121")
    print("-  alexnet")
    quit()

if(not(args.lrn > 0 and args.lrn < 1)):
    print("Please enter a valid learn rate between 0 and 1 (exclusive of 0 and 1)")
    quit()

if(args.hidden_units <= 0):
    print("Please enter a valid hidden unit value greater than 0. Default is 1024")                    
    quit()          
         
if(args.epochs <= 0):
    print("Please enter a valid number of epochs greater than 0. Default is 3")                    
    quit()    

if args.GPU not in ['cpu', 'GPU']:
    print("Please enter 'cdu' or 'GPU' for the device")                    
    quit()    

if(args.batch_size <= 0):
    print("Please enter a batch size greater than 1")
                    

#apply variable names to args
save_dir = args.save_dir +'/'
arch = args.arch
lrn = args.lrn
hidden_units = args.hidden_units
epochs = args.epochs
GPU = args.GPU
batch_sz = args.batch_size

#setup data directories
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# GPU argument
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

# updated GPU as explained in review 2/7/2021
if device and torch.cuda.is_available():
   device = torch.device('cuda')
else:
   device = torch.device('cpu')
   
   
   
#Define transforms for training, validation, and testing (updated per review 2/7/2021)
training_transforms = transforms.Compose([transforms.RandomRotation(15),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

validation_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

testing_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

   
#Load the training datasets with ImageFolder
training_datasets = datasets.ImageFolder(data_dir, transform = training_transforms)
validation_datasets = datasets.ImageFolder(valid_dir, transform = validation_transforms)
testing_datasets = datasets.ImageFolder(test_dir, transform = testing_transforms)

#Define thetraining  dataloaders
training_dataloader = torch.utils.data.DataLoader(training_datasets, batch_size = batch_sz, shuffle = True)
validation_dataloader = torch.utils.data.DataLoader(validation_datasets, batch_size = batch_sz)
testing_dataloader = torch.utils.data.DataLoader(testing_datasets, batch_size = batch_sz)
                    

#return model from argument ref https://knowledge.udacity.com/questions/479950
model = getattr(models,arch)(pretrained = True)

#update on returning models from argument (suggensted method)
'''def select_model(arch):
    t_models = {
                'vgg11': models.vgg11(pretrained = True),
                'vgg13': models.vgg13(pretrained = True),
                'vgg16': models.vgg16(pretrained = True),
                'vgg19': models.vgg19(pretrained = True),
                'densenet121': models.densenet121(pretrained = True),
                'alexnet': models.alexnet(pretrained = True)}

    model = t_models.get(arch)

    classifier = None
    optimizer = None

    if arch == 'vgg11' or arch == 'vgg13' or arch == 'vgg16' or arch == 'vgg19':
        classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(25088,hidden_units)),
                ('relu',nn.ReLU()),
                ('dropout',nn.Dropout(p = 0.2)),
                ('fc2',nn.Linear(hidden_units,102)),
                ('output',nn.LogSoftmax(dim = 1))]))
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr = lrn)
    elif arch == 'densenet121':
        classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(1024,hidden_units)),
                ('relu',nn.ReLU()),
                ('dropout',nn.Dropout(p = 0.2)),
                ('fc2',nn.Linear(hidden_units,102)),
                ('output',nn.LogSoftmax(dim = 1))]))
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr = lrn)
    elif arch == 'alexnet':
        classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(9216,hidden_units)),
                ('relu',nn.ReLU()),
                ('dropout',nn.Dropout(p = 0.2)),
                ('fc2',nn.Linear(hidden_units,102)),
                ('output',nn.LogSoftmax(dim = 1))]))
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr = lrn)'''

 
#Turn off gradients
for param in model.parameters():
    param.requires_grad = False 
    
#replace classifier
classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088,hidden_units)),
            ('relu',nn.ReLU()),
            ('dropout',nn.Dropout(p = 0.1)),
            ('fc2',nn.Linear(hidden_units,102)),
            ('output',nn.LogSoftmax(dim = 1))]))
model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = lrn)
model.to(device)           

print(model)

# Prints out training loss, validation loss, and validation accuracy as the network trains
steps = 0
running_loss = 0
print_every = 10

#old incorrect method for epoch in range(epochs):
'''    for images, labels in training_dataloader:
        steps += 1
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval()
            test_loss = 0
            accuracy = 0
            
            for images, labels in training_dataloader:
                images, labels = images.to(device), labels.to(device)
                
                logps = model(images)
                loss = criterion(logps, labels)
                test_loss += loss.item()
                
                #calculate accuracy
                with torch.no_grad():
                    ps = torch.exp(logps)
                    top_ps, top_class = ps.topk(1, dim = 1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                
                print(f"Epoch {epoch + 1}/{epochs}.."
                      f"Train loss: {running_loss/print_every:.3f}.."
                      f"Test loss: {test_loss/len(training_dataloader):.3f}.."
                      f"Test accuracy: {accuracy/len(training_dataloader):.3f}..")
                
                running_loss = 0
'''
#updated correct method 2/8/2021                
for epoch in range(epochs):
    for images, labels in training_dataloader:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            validation_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for images, labels in validation_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    logps = model.forward(images)
                    loss = criterion(logps, labels)
                    
                    validation_loss += loss.item()
                    
                    ps = torch.exp(logps)
                    top_ps, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Valid loss: {validation_loss/len(validation_dataloader):.3f}.. "
                  f"Valid accuracy: {accuracy/len(validation_dataloader):.3f}")
            running_loss = 0
            model.train()


# save the model. Note: add'l info from Dr. Johan's question (https://knowledge.udacity.com/questions/286164)
model.state_dict().keys()
model.class_to_idx = training_datasets.class_to_idx
checkpoint = {'class_to_idx':model.class_to_idx,
              'state_dict':model.state_dict(),
              'arch':arch,
              #'input_size':25088,
              'output_size':102,
              'epochs':epochs,
              'dropout': 0.2,
              'classifier':model.classifier,
              'optimizer_dict':optimizer.state_dict(),
              'learning_rate':lrn}

torch.save(checkpoint, save_dir + 'checkpoint.pth')