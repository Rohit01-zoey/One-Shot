from data.asl import ASL
from model.resnet import ResNetV2

import numpy as np
import random
import torch
import torchvision.transforms as transforms
import os
import argparse

def main(args):
    global CUDA
    CUDA = "cuda:"+args.cuda
    
    dataset = ASL(root = f'./dataset/{args.dataset}', type = 'train', transforms=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]), transforms.Resize((32,32))]))
    loader = dataset.load()
    
    depth = 56
    in_planes = 3
    num_classes = dataset.num_classes
    data_augmentation = False
    model = ResNetV2(depth, in_planes, num_classes, data_augmentation).to(CUDA)
    
    dl = torch.utils.data.DataLoader(loader, batch_size=128, shuffle=True)
    print('please wait')
    
    criterion_CE = torch.nn.CrossEntropyLoss().to(CUDA)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    print(len(dl))
    
    for epoch in range(100):
        train_loss = 0
        train_acc = 0
        for input, label in dl:
            input, label = input.to(CUDA), label.to(CUDA)
            output = model(input)
            output_idx = torch.argmax(output, dim=1)
            train_acc += torch.sum(output_idx == label)/len(input)
            loss = criterion_CE(output, label)
            train_loss += loss.item()
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                  
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}, Accuracy: {torch.sum(output_idx == label)/len(input)}')
        




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type = str, default = 'ASL', help = 'dataset name')
    parser.add_argument('--cuda', '-c', type=str, default='0', help='cuda device id')
    
    
    args = parser.parse_args()
    main(args)