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
from os import listdir
from os.path import isfile, join

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
    correct_mask_matches=0
    incorrect_mask_matches=0
    correct_nomask_matches=0
    incorrect_nomask_matches=0
    mask_files = [f for f in listdir('masks/Train/WithMask') if isfile(join('masks/Train/WithMask', f))]
    withoutmask_files = [f for f in listdir('masks/Train/WithoutMask') if isfile(join('masks/Train/WithoutMask', f))]
    
    # for i in range(len(mask_files)):
    #     image = cv2.imread(join('masks/Train/WithMask',mask_files[i]))
    #     label=classify_face(image)
    #     if label=='with_mask':
    #         correct_mask_matches+=1
    #     else:
    #         incorrect_mask_matches+=1
    #     print("mask")
    
    for i in range(len(withoutmask_files)):
        image = cv2.imread(join('masks/Train/WithoutMask',withoutmask_files[i]))
        label=classify_face(image)
        if label=='without_mask':
            correct_nomask_matches+=1
        else:
            incorrect_nomask_matches+=1
        print("nomask")

    # labels = 'Incorrect Label - Mask', 'Correct Label - Mask'
    # sizes = [incorrect_mask_matches,correct_mask_matches]
    # explode = (0, 0.1)  # only "explode" the 2nd slice 

    # fig1, ax1 = plt.subplots()
    # ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
    #         shadow=True, startangle=90)
    # ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # plt.show()

    labels = 'Incorrect Label- No Mask', 'Correct Label- No Mask'
    sizes = [incorrect_nomask_matches,correct_nomask_matches]
    explode = (0, 0.1)  # only "explode" the 2nd slice 

    fig2, ax2 = plt.subplots()
    ax2.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()
    

    



