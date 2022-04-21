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
import ssl

exp ='observations/experiements/dest_folder/'
dat = 'observations/experiements/data/'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]), 
    'test' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
def train_pth(exp, dat, typ):
    if typ == 'train':
        file_name = 'train.csv'
    elif typ == 'test':
        file_name = 'test.csv'
    else:
        print("incorrect path")
        exit()
    file_path = os.path.join(exp, file_name)
    train_df = pd.read_csv(file_path, delimiter=',')
    files = []
    fonts = []
    for row in train_df.iterrows():
        files.append(os.path.join(dat, row[1]['class'], row[1]['filename']))
        fonts.append(row[1]['class'])
    
    return files, fonts

def copy_imgs(file_path, file_class, destination_dir):
    font_folder = os.path.join(destination_dir, file_class)
    if os.path.exists(font_folder) == False:
        os.makedirs(font_folder)
    
    print("File being copied from {}:{}".format(file_path, font_folder))
    shutil.copy(file_path, font_folder)
    #shutil.copyfile(file_path, font_folder)


def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()
    best_acc = 0.0
    best_model = copy.deepcopy(model.state_dict())
    
    new_freeze_state = None
    prev_freeze_state = False
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print('-' * 10)
        
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
        
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc:{:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())
            
            print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))
    
    model.load_state_dict(best_model)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    #fig = plt.figure(figsize=(10,10))
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            print(preds,"predicitons")
            
            
            for j in range(inputs.size()[0]):
                images_so_far +=1
                #ax = plt.subplot(num_images//len(labels)-1, len(labels), images_so_far)
                #ax.axis('off')
                #ax.set_title('true: {} predicted: {}'.format(class_names[labels[j]], class_names[preds[j]]))
                print('true: {} predicted: {}'.format(classes[labels[j]], classes[preds[j]]))
                #imshow(inputs.cpu().data[j])
                
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

if __name__=='__main__':
    X_train, y_train = train_pth(exp, dat, typ='train')
    X_test, y_test = train_pth(exp, dat, typ='test')

    train_dir = os.path.join(exp, 'train')
    test_dir = os.path.join(exp, 'test')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for file_path, font_class in zip(X_train, y_train):
        copy_imgs(file_path, font_class, train_dir)


    image_datasets = {typ: datasets.ImageFolder(os.path.join(exp, typ), data_transforms[typ]) for typ in ['train', 'test']}
    dataloaders = {typ: torch.utils.data.DataLoader(image_datasets[typ], 
                                                batch_size=16, 
                                                shuffle=True, 
                                                num_workers=2) 
                for typ in ['train', 'test']}
    classes=image_datasets['train'].classes
    dataset_sizes= {typ:len(image_datasets[typ]) for typ in ['train','test']}


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    ssl._create_default_https_context = ssl._create_unverified_context  
    model_ft = models.resnet101(pretrained=True)

    num_frts = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_frts, len(classes))

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
    optimizer_ft = optim.Adagrad(model_ft.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
    torch.save(model_ft, '/content/mask1_model_resnet101.pth')
    torch.save('C:Users/bierm/OneDrive/Desktop/mask1_model_resnet101.pth')

    visualize_model(model_ft)