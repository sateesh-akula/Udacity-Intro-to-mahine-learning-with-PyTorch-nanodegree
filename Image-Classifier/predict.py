import torch
import torch.nn.functional as F
from torch import nn,optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import numpy as np
from PIL import Image
import argparse
import json
from tabulate import tabulate

import matplotlib.pyplot as plt

%matplotlib inline

parser = argparse.ArgumentParser(description ='Process training input')
parser.add_argument('input', help='Path to input image', type=str)
parser.add_argument('checkpoint', help='Path to checkpoint file', type=str)
parser.add_argument('-k', '--top_k', help='Top K classes to display. Default top_k=1', type=int, default=3)
parser.add_argument('-n', '--category_names', help='Display category name. Default category_names = cat_to_name.json', default='cat_to_name.json')
parser.add_argument('-gpu', '--gpu', help='Use GPU', action='store_true')


def main():
    global arg, device
    arg = parser.parse_args()
    
    if arg.gpu:
        print('Using GPU')
        device='cuda'
    else:
        devive='cpu'
        
    arg = parser.parse_args()
    
    image_path = arg.input
    file = arg.checkpoint
    
    model = load_checkpoint(file)
    image = Image.open(image_path)
    pr_image = process_image(image)
    
    ps, idx = predict(image_path, model, arg.top_k)
    
    with open(arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    classes = [cat_to_name[i] for i in idx]
    
    if arg.top_k==1:
        print('\n\n Flower class: ', classes[0],'(index=',idx[0],') with probability', '{0.3f}'.format(ps[0]*100) + '% \n')
    else:
        print('\n| ------------------------------')
        print('\n| The top K', arg.top_k, 'classes are:')
        print('\n| --------------------')
        line=[]
        
        for i in range(arg.top_k):
            line.append([classes[i], idx[i], '{0.3f}'.format(ps[i]*100) + '%'])
        try:
            print(tabulate(line, headers=['name','Index','probability'], tablefmt='orgtbl'))
        except:
            print('warning: Package tabulate was not installed. Requires tabulate to display results in table. Please install tabulate')
            print('name', 'Index', 'probability')
            [print(i) for i in line]
   
def load_checkpoint(file):
    chkpt = torch.load(file,map_location='cpu')
    arch = chkpt['arch']
    
    if arch =='vgg13':
        model = models.vgg13(pretrained=True)
        
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    base = 256        # Resize
    width, height = image.size
    if width <=height:
        nw_height = int(float(height/width)*base)
        pr_image = image.resize((base, nw_height), Image.ANTIALIAS)
    else:
        nw_width = int(float(width/height)*base)
        pr_image = image.resize((base, nw_width), Image.ANTIALIAS)
        
    #Crop Center Image
    width, height = pr_image.size
    left = (width - 224)/2
    right = (width + 224)/2
    top = (height - 224)/2
    bottom = (height+224)/2
    pr_image = pr_image.crop((left,top,right,bottom))
                             
    # value convertion 0-1
    np_img = np.array(pr_image)/255 
                             
    # normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = np.divide(np_img-mean, std)
                             
    transpose_image = np_img.transpose((2,0,1))
                             
    return transpose_image

def predict(image_path, model, topk=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = Image.open(image_path)
    pr_image = process_image(image)
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    model.eval()
    
    with torch.no_grad():
        
        torch_image = torch.from_numpy(pr_image).float()
        torch_image.unsqueeze_(0)
        torch_image = torch_image.to(device)
        
        outputs = model.forward(torch_image)
        ps, idx = outputs.topk(topk)
        ps = ps.cpu()
        ps = np.exp(ps.numpy())
        
        inverse_map = {v: k for k, v in model.class_to_idx.items()}
        idx = idx.cpu()
        idx = [inverse_map[i] for i in idx.numpy()[0]]
#         classes = [cat_to_name[i] for i in idx]
#         print("Probabilities : ", ps[0])
#         print('Classes: ', idx)
#         print('Class_name: ', classes)
        
    return ps[0], idx

if __name__ == '__main__':
    main()
