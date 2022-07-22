import torch
import torch.nn.functional as F
from torch import nn,optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import numpy as np
from PIL import Image
import argparse

import matplotlib.pyplot as plt

%matplotlib inline


parser = argparse.ArgumentParser(description='Training Input')
parser.add_argument('data_dir', help='Image Directory', type=str)
parser.add_argument('-sd', '--save_directory', help=' Directory to save, checkpoint file', type=str, default='.')
parser.add_argument('-arch', '--arch', help='Choose architecture vgg13 or other. Default arch=vgg13', default='vgg13')
parser.add_argument('-r', '--learning_rate', help='Learning rate. Default learning_rate = 0.01', default=0.01, type=float)
parser.add_argument('-e', '--epochs', help='Number of Epochs. Default=20', type=int, default=20)
parser.add_argument('-u', '--hidden_units', help='Number of hidden units. Default=512', type=int, default=512)
parser.add_argument('-gpu', '--gpu', help='Use GPU', action='store')

def main():
    global arg, device
    arg = parser.parse_args()
    data_dir = arg.data_dir
    file = arg.save_directory + '/' + 'chk_' + arg.arch + '.pth
    learning_rate = arg.learning_rate
    hidden_units = arg.hidden_units
    epochs = arg.epochs
    
    if arg.gpu:
        print('Use GPU')
        device = 'cuda'
    else:
        device = 'cpu'
        
    if arg.arch = 'vgg13':
        print('Initializing vgg13 model')
        model = model.vgg13(pretrained=True)
        in_features = 25088
    else:
        print('warning: ', arg.arch, 'Missing model selection, Will use vgg13 as default')
        print('Initializing vgg13 model')
        in_features = 25088

    train_transforms = transforms.Compose([transforms.RandomRotation(90), transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(0.2), transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # TODO: Load the datasets with ImageFolder
    train_imagedata = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_imagedata = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
    test_imagedata = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    
    # Define Dataloaders
    trainloader = torch.utils.data.DataLoader(train_imagedata, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_imagedata, batch_size = 64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_imagedata, batch_size = 64, shuffle = True)
    
    #Build & Train network
    #setting requires_grad Parameter to off
    for param in vgg_ml.parameters():
        param.requires_grad = False

    #new classifier
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, 1000)),
                            ('relu', nn.ReLU()),
                            ('dp1', nn.Dropout(0.2)),
                            ('fc2', nn.Linear(1000,102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    test_run(model, trainloader, validloader, criterion, optimizer, epochs, 20)
    test_accuracy(model, testloader)
    checkpoint(model, train_imagedata,file)
    
    def validation(model, validloader, criterion):
        valid_loss = 0
        accuracy = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        for images, labels in validloader:
            images, labels = images.to(device), labels.to(device)

            output = model.forward(images)
            valid_loss += criterion(output, labels).item()

            ps = torch.exp(output)
            eqty = (labels.data == ps.max(dim=1)[1])
            accuracy += eqty.type(torch.FloatTensor).mean()

        return valid_loss, accuracy
    
    def test_run(model, trainloader, validloader, criterion, optimizer, epochs, min_times=20):
        epochs = epochs
        min_times = min_times
        steps = 0

    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        for e in range(epochs):
            run_loss = 0
            for i, (images, labels) in enumerate(trainloader):
                steps += 1
                images, labels = images.to(device), labels.to(device) 
                optimizer.zero_grad()

                outputs = model.forward(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                run_loss += loss.item()

                if steps % min_times == 0:
                    model.eval()

                    with torch.no_grad():
                        valid_loss, accuracy = validation(model, validloader, criterion)

                    print('Epochs: {}/{}.. '.format(e+1, epochs),
                          'Training loss: '+str(run_loss/min_times),
                          'Validation loss: '+str(valid_loss/len(validloader)),
                          'Validation accuracy: '+str(accuracy/len(validloader)))
                    run_loss = 0
                    model.train()
                
    def test_accuracy(model, testloader):
        success = 0
        total = 0
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _,predicted = torch.max(outputs.data,1)
                total += labels.size(0)
                success += (predicted == labels).sum().item()
        print('Total Images: 'total)
        print('Accuracy of the test images: '+str((success/total)*100)+'%')
    
    def checkpoint(model, train_imagedata, file):
        model_file = file
        model.class_to_idx = train_imagedata.class_to_idx
        print('Saving checkpoint file to', file)

        checkpoint_dict = {
            'model_arch':'vgg19',
            'classifier': model.classifier,
            'class_to_idx': model.class_to_idx,
            'state_dict': model.state_dict()
        }

        torch.save(checkpoint_dict, model_file)
    
    if __name__ == '__main__':
        main()
    
    