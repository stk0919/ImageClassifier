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
# Total 7 arguments: data_dir, --save_dir, --arch, --learning_rate, --hidden_units, --epochs, --gpu 

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', type = str, metavar = '', help = 'Directory of files. Required.')
parser.add_argument('--save_dir', type = str, metavar = '', help = 'Save directory. Default is Checkpoint', default = 'Checkpoint')
parser.add_argument('--arch', type = str, metavar = '', help = 'Choose vgg16 or vgg13. Default is vgg16', default = 'vgg16')
parser.add_argument('--lrn', type = float, metavar = '', help = 'Learn rate. Default is 0.002', default = 0.002)
parser.add_argument('--hidden_units', type = int, metavar = '', help = 'Hidden units. Default is 1024', default = 1024)
parser.add_argument('--epochs', type = int, metavar = '', help = 'Number of epochs. Default is 3', default = 3)
parser.add_argument('--GPU', type = str, metavar = '', help = 'Use GPU/cuda. Default is GPU/cuda', default = 'GPU')

args = parser.parse_args()

#apply variable names to args
save_dir = args.save_dir
arch = args.arch
lrn = args.lrn
hidden_units = args.hidden_units
epochs = args.epochs
GPU = args.GPU


#setup data directories
data_dir = args.data_dir

# load cat_to_name
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# GPU argument
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

    
#Define transforms for training
training_transforms = transforms.Compose([transforms.RandomRotation(15),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

#Load the training datasets with ImageFolder
training_datasets = datasets.ImageFolder(data_dir, transform = training_transforms)

#Define thetraining  dataloaders
training_dataloader = torch.utils.data.DataLoader(training_datasets, batch_size = 64, shuffle = True)

#return model from argument ref https://knowledge.udacity.com/questions/479950 Thank you Arun!
model = getattr(models,arch)(pretrained = True)

#Turn off gradients
for param in model.parameters():
    param.requires_grad = False 

#replace classifier
classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088,hidden_units)),
            ('relu',nn.ReLU()),
            ('dropout',nn.Dropout(p = 0.2)),
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
print_every = 20

for epoch in range(epochs):
    for images, labels in training_dataloader:
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


# save the model
model.state_dict().keys()
model.class_to_idx = training_datasets.class_to_idx
checkpoint = {'class_to_idx':model.class_to_idx,
              'state_dict':model.state_dict(),
              'arch':arch,
              'input_size':25088,
              'output_size':102,
              'epochs':epochs,
              'dropout': 0.2,
              'classifier':model.classifier,
              'optimizer_dict':optimizer.state_dict(),
              'learning_rate':lrn}

torch.save(checkpoint, save_dir + '/checkpoint.pth') 
