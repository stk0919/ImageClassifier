# prediction - continue from train.py
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

#Argparse items:
#1. Return top K most likely classes: python predict.py input checkpoint --top_k 3
#2. Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
#3. Use GPU for inference: python predict.py input checkpoint --gpu
# Mandatory: path/to/image, checkpoint


parser = argparse.ArgumentParser()

parser.add_argument('path_to_image', type = str, metavar = '', help = 'Path to image and file name')
parser.add_argument('checkpoint', type = str, metavar = '', help = 'Path and file name for model')
parser.add_argument('--top_k', type = int, metavar = '', help = 'Top K most likely classes. Default is 5', default = 5)
parser.add_argument('--category_names', type = str, metavar = '', help = 'Use mapping of categories to real names. Default is cat_to_name.json', default = 'cat_to_name.json')
parser.add_argument('--GPU', type = str, metavar = '', help = 'Use GPU/cuda. Default is GPU/cuda', default = 'GPU')


args = parser.parse_args()

#apply variable names to args
image_dir = args.path_to_image
input_chk = args.checkpoint
top_k = args.top_k
category_names = args.category_names
GPU = args.GPU


#GPU argument
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

if device and torch.cuda.is_available():
   device = torch.device('cuda')
else:
   device = torch.device('cpu')    
    
     
    
#load category file
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    
#load checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    #return model from argument ref https://knowledge.udacity.com/questions/479950 Thank you Arun!    
    model = getattr(models,arch)(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    learning_rate = checkpoint['learning_rate']
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    return model

model = load_checkpoint(input_chk)

'''def load_model(checkpoint):
    t_models = {
            'vgg11': models.vgg11(pretrained = True),
            'vgg13': models.vgg13(pretrained = True),
            'vgg16': models.vgg16(pretrained = True),
            'vgg19': models.vgg19(pretrained = True),
            'densenet121': models.densenet121(pretrained = True),
            'alexnet': models.alexnet(pretrained = True)}

    model = t_models.get(checkpoint['arch'])
    if checkpoint['arch'] == 'vgg11' or checkpoint['arch'] == 'vgg13' or checkpoint['arch'] == 'vgg16' or checkpoint['arch'] == 'vgg19' or checkpoint['arch'] == 'densenet121':
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
    else:
        model.fc = checkpoint['fc']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
    return model

load_model(input_chk)'''

# Process image
def process_image(image):
    #Scales, crops, and normalizes a PIL image for a PyTorch model,
        #returns an Numpy array
    #
    pil_image = Image.open(image)

    image_transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),      
                                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
    new_image = image_transform(pil_image)
    return new_image  
    
    
def predict(image_path, model, topk=5):
    # #Predict the class (or classes) of an image using a trained deep learning model.
    #
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
    
    #get cat_to_names
    flower_names = [cat_to_name[str(i)] for i in classes]
    
    probs = [np.around(n, 2) for n in probs]
    
    #combine flower names with probabilities and print 
    flower_result = (zip(flower_names, probs))
    for key, value in flower_result:
        print('Flower: {:20s} Prob %: {:4.1f}%'.format(key.title(), round(value*100,2)))
      

predict(image_dir, model, top_k)

#test values
#image_path = 'flowers/test/10/image_07104.jpg' <- globe thistle
#image_path = 'flowers/test/1/image_06743.jpg' <- pink primrose