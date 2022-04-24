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
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import time
import copy
from PIL import Image, ImageOps
import glob
import cv2
from PIL import ImageFont
from PIL import ImageDraw 


filepath = 'mask1_model_resnet101.pth'
model = torch.load(filepath)


class_names = ['with_mask',
 'without_mask'
]



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    #pil_image = Image.open(image)
    pil_image = image
   
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = image_transforms(pil_image)
    return img
    
    


def classify_face(image):
    device = torch.device("cpu")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #im_pil = image.fromarray(image)
    #image = np.asarray(im)
    im = Image.fromarray(image)
    image = process_image(im)
    print('image_processed')
    img = image.unsqueeze_(0)
    img = image.float()

    model.eval()
    model.cpu()
    output = model(image)
    print(output,'##############output###########')
    _, predicted = torch.max(output, 1)
    print(predicted.data[0],"predicted")


    classification1 = predicted.data[0]
    index = int(classification1)
    print(class_names[index])
    return class_names[index]



if __name__ == '__main__':
    #map_location=torch.device('cpu')
    image = cv2.imread('test_img_joe.jpg')
    label = classify_face(image)
    print("the label is", label)
    if label=='with_mask':
        color='green'
    else:
        color='red'
    
   

    height,width=image.shape[:2]
    il=Image.open('test_img_joe.jpg')
    il= ImageOps.expand(il,border=20,fill=color)
    draw = ImageDraw.Draw(il)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype(r'C:\Users\System-Pc\Desktop\arial.ttf', 90)
    # draw.text((x, y),"Sample Text",(r,g,b))
    draw.text((0,0),label,(255,255,255),font=font)
    
    il.save('labeled_output.jpg')



