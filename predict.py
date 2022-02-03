import argparse
import numpy as np
import torch
import json

from collections import OrderedDict
 
from torch import nn
from torch import optim
from torch.autograd import Variable 
from torchvision import models
import torch.nn.functional as F

from train import construct_model
from PIL import Image





def get_command_line_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image', type=str, help='path to the image to test')
    parser.add_argument('--topk', type=int, help='Top classes to return',             default=5)
    parser.add_argument('--checkpoint', type=str, help='Saved Checkpoint') 
    parser.add_argument('--gpu', default='False',action='store_true', help='Where to use gpu or cpu')
    parser.add_argument('--epoch', type=int, help='amount of times the model will train')
    parser.add_argument('--labels', type=str, help='file for label names',             default='paind-project/cat_to_name.json')
    # arch and hidden units of checkpoint added per review
    parser.add_argument('--arch', type=str, default='vgg16', help='chosen model')
    parser.add_argument('--hidden_units', type=int, default='4000', help='hidden units for the model')

    return parser.parse_args()

def load_checkpoint(checkpoint, arch, hidden_units):
    
      
      # Credit to Michael for providing me with a way to convert gpu tenors to cpu 
    checkpoint_state = torch.load(checkpoint, map_location = lambda storage, loc: storage)

    class_to_idx = checkpoint_state['class_to_idx']

   
    model, optimizer, criterion = construct_model(hidden_units, class_to_idx,arch)

    
    model.load_state_dict(checkpoint_state['state_dict'])
    optimizer.load_state_dict(checkpoint_state['optimizer'])

    
    print("Loaded checkpoint => {} with arch {}, hidden units {} and epochs {}".format
          (checkpoint, 
           checkpoint_state['arch'], 
           checkpoint_state['hidden_units'], 
           checkpoint_state['epochs']))
    return model, optimizer, criterion


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
     
   
    size = 256,256
    
    pill_image = Image.open(image)
    pill_image = pill_image.resize(size)
    pill_image = pill_image.crop((16,16,240,240))
    
    np_image = np.array(pill_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = ((np_image/255) -  mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
            
    return np_image




def predict(image, checkpoint, topk, labels, arch, hidden_units,device, gpu=False):
    model, optimizer, criterion = load_checkpoint(checkpoint, arch, hidden_units)
    
    model.eval()
    
    image = process_image(image)
    if gpu:
        myInput = torch.FloatTensor(image).cuda()
    else:
        myInput = torch.FloatTensor(image)
    model= model.to(device)
    myInput.unsqueeze_(0)
    output = model(myInput)
    ps = torch.exp(output)
    probs, classes = torch.topk(ps, topk)
    inverted_class_to_index = {model.class_to_idx[x]: x for x in model.class_to_idx}
    new_classes = []
    
    for index in classes.cpu().numpy()[0]:
        new_classes.append(inverted_class_to_index[index])
        
    return probs.cpu().detach().numpy()[0], new_classes 
def main():
    args = get_command_line_args()
    

    use_gpu = torch.cuda.is_available() and args.gpu
    if use_gpu:
        print("Using GPU.")
        device = torch.device('cuda')
    else:
        print("Using CPU.")
        device = torch.device('cpu')
    if(args.checkpoint,args.image):
        probs, new_classes = predict(args.image, args.checkpoint, args.topk,args.labels, args.arch, args.hidden_units ,device,args.gpu)
    with open(args.labels, 'r') as j:
# Thanks to michael for helping me understand the use of argmax
        cat_to_name = json.load(j)
    biggest_idx = np.argmax(probs)   
    max_class = new_classes[biggest_idx]
    first = cat_to_name[max_class]
    print("---------Classes and Probabilities---------")
    for i, idx in enumerate(new_classes):
        print("Class:", cat_to_name[idx], "Probability:", probs[i])
main()
    