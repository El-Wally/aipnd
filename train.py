import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data


import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import sys
import time
import copy
import argparse
from collections import OrderedDict 

# Define command line arguments
def get_command_line_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='paind-project/flowers',help='Images Folder')
    parser.add_argument('--gpu', action='store_true',default='False', help='Use GPU if available')
    parser.add_argument('--arch', default='vgg16',type=str, help='Model architecture')
    parser.add_argument('--hidden_units', type=int, default='4000',help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default='10',help='Number of epochs to use')
    parser.add_argument('--learning_rate', type=float, default='0.001',help='Learning rate')  
    parser.add_argument('--checkpoint', type=str, default='',help='Save trained model to file')
 
    
    return parser.parse_args()

  



def data_loaders():
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
        
    validation_transforms =  transforms.Compose([transforms.Resize(256),   
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),    
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])
    testing_transforms =  transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    data_dir = 'paind-project/flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    image_datasets = {
            'training': datasets.ImageFolder(train_dir, transform=train_transforms),
            'validation': datasets.ImageFolder(valid_dir, transform=validation_transforms),
            'testing': datasets.ImageFolder(test_dir, transform=testing_transforms)
        }
    data_loader = {
            'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size = 64, shuffle=True),
            'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size = 64, shuffle=True),
            'testing': torch.utils.data.DataLoader(image_datasets['testing'], batch_size = 32, shuffle=False)
        }
    label_number = len(image_datasets['training'])
    class_to_idx = image_datasets['training'].class_to_idx
    
    return data_loader,class_to_idx,



   
   


def construct_model(hidden_units,class_to_idx,arch='vgg16',   learning_rate = 0.001):

    
                   
    if arch == 'densenet121':
         model = models.densenet121(pretrained=True)
         input_size = model.classifier.in_features
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_size = model.classifier[0].in_feature
    
    else:
        raise TypeError("The arch specified is not supporte")
     
 # Freezing parameters so we don't backpropagate through them 
    for param in model.parameters():
        
        param.requires_grad = False

    output_size = 102     
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units,output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.class_to_idx = class_to_idx
    criterion = nn.NLLLoss()
    
    
    return model, optimizer, criterion 
# This method trains a model
def train_model(image_datasets, arch, hidden_units, epochs, learning_rate,data_load,class_to_idx, gpu=True, checkpoint=''):
    # Use command line values when specified
    

         
        
    model, optimizer, criterion = construct_model(hidden_units,class_to_idx,  arch, learning_rate)
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        model.cuda()
        
    else:
        device = torch.device("cpu")
    print('Network architecture:', arch)
    print('Number of hidden units:', hidden_units)
    print('Number of epochs:', epochs)
    print('Learning rate:', learning_rate)

    # Use gpu if selected and available
    if gpu and torch.cuda.is_available():
        print('Using GPU for training')
        device = torch.device("cuda")
        model.cuda()
    else:
        print('Using CPU for training')
        device = torch.device("cpu")     
    sys.stdout.write("Training")
    steps = 0
    acc = 0
    epochs = 10
    print_every = 40
    time_start = time.time()
    for e in range(epochs):
        
        running_loss = 0 
        running_loss1 = 0
        for ii, (inputs, labels) in enumerate(data_load['validation']):
            model.train()
            steps += 1 
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() 
            if ii % 5 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
                running_loss +=  loss.item()
            

            if steps % print_every == 0:
                
                model.eval()
                for i, (inputs, labels) in enumerate(data_load['testing']): 
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    output = model.forward(inputs)
                    
                    running_loss1 += criterion(output, labels).item()
                print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every), 
                     "Validation Loss: {:.3f}".format(running_loss1/len(data_load['testing'])))
                
            
            
                print()
                sys.stdout.write("Training")
                running_loss = 0 
                running_loss1 = 0
    time_end = time.time()-time_start
    print("Training is now complete!")
    
    print("\n** Total Elapsed Runtime:",
          str(int((time_end/3600)))+":"+str(int((time_end%3600)/60))+":"
          +str(int((time_end%3600)%60)) )
    x = accuracy(data_load,model)
    return model, optimizer, criterion
    
def accuracy(data_load,model):
    print('Testing start')
    model.to('cuda')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_load["testing"]:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
def save_checkpoint(arch, learning_rate, hidden_units, epochs, save_path, optimizer,criterion,class_to_idx):
    
    state = {
        'arch': arch,
        'learning_rate': learning_rate,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'state_dict': arch.state_dict(),
        'optimizer': optimizer.state_dict(),
        'criterion': criterion.state_dict(),
        'class_to_idx': class_to_idx
    }

    torch.save(state, save_path)
    
    print("Checkpoint saved to {}".format(save_path))
def main():
    
    
    
        
    args = get_command_line_args()
   
    data_load , class_to_idx= data_loaders()
    print(data_load['training'])
    model,optimizer,criterion =  train_model(args.dir,args.arch,  args.hidden_units, 
                   args.epochs, args.learning_rate,data_load,class_to_idx,args.gpu,args.checkpoint )
# adviced by curtis to know if user did input checkpoint destination or no    
    if len(args.checkpoint) == 0:
        args.checkpoint = args.checkpoint + 'checkpoint.pth'
    else:
        args.checkpoint = args.checkpoint + '/checkpoint.pth'
    save_checkpoint(model,args.learning_rate,args.hidden_units,
                   args.epochs,args.checkpoint,optimizer,criterion,class_to_idx)
#Found it from forum to avoid training when calling construct_model function from predict.py
if __name__ == "__main__":
    main()

