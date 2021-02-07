# prediction - continue from train.py (add argparse for arguments - load checkpoint)
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

#Options:
#1. Return top K most likely classes: python predict.py input checkpoint --top_k 3
#2. Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
#3. Use GPU for inference: python predict.py input checkpoint --gpu

parser = argparse.ArgumentParser()

parser.add_argument('image_dir', type = str, metavar = '', help = 'Path to image and file name')
parser.add_argument('--top_k', type = int, metavar = '', help = 'Top K most likely classes. Default is 5', default = 5)
parser.add_argument('--category_names', type = str, metavar = '', help = 'Use mapping of categories to real names. Default is cat_to_name.json', default = 'cat_to_name.json')
parser.add_argument('--GPU', type = str, metavar = '', help = 'Use GPU/cuda. Default is GPU/cuda', default = 'GPU')

args = parser.parse_args()

#apply variable names to args
image_dir = args.image_dir
top_k = args.top_k
category_names = args.category_names
GPU = args.GPU


#GPU argument
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    model = getattr(models,arch)(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    learning_rate = checkpoint['learning_rate']
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    return optimizer, model

model = load_checkpoint('Checkpoint/checkpoint.pth')
model

  
# Process image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)

    image_transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),      
                                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
    new_image = image_transform(pil_image)
    return new_image  


    
    
    
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
   #cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)    
    model.eval()
    
    image = process_image(image_path)

    image = image.unsqueeze_(0)
    
    with torch.no_grad():
        inputs = image.to(device)
        output = model.forward(inputs)

# referenced question from Amber M (https://knowledge.udacity.com/questions/335109)
    prediction = F.softmax(output.data, dim = 1)
    
    probs, indices = prediction.topk(topk)
    probs = probs.cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]
    
    idx_to_class = {value:key for key, value in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in indices]
        
    return probs, classes

#test
image_path = 'flowers/test/10/image_07104.jpg'

predict(image_path, model)

#flower_names = [cat_to_name[str(i)] for i in classes]
#print(flower_names)
#print(flower_names[0])