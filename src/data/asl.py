'''loads the ASL dataset in ~/rohit/data/ASL/ASL_alphabet_train/ and ~/rohit/data/ASL/ASL_alphabet_test/'''

import os
import torch
import torchvision
import torchvision.transforms as transforms


class ASL:
    '''Loads the ASL dataset'''
    
    def __init__(self, root, type = 'both', transforms = None) -> None:
        """Loads the ASL dataset

        Args:
            root (str): The path to the root directory of the dataset
            type (str, optional): Whether to load the train set, test set or both. Possible options 'train'|'test'|'both'. For the first two a Dataset object is returned
            If 'both' is selected, a dicitionary containing the two Dataset objects with keys 'train' and 'test'. Defaults to 'both'.
            transforms (torchvision.transforms, optional): Transforms to be applied to the dataset. Defaults to None.
        """
        self.root = root
        self.type = type
        self.train_transform = transforms['train'] if isinstance(transforms, dict) else transforms
        self.test_transform = transforms['test'] if isinstance(transforms, dict) else transforms
        self.num_classes = 29
        self.classes  = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 
                         'X', 'Y', 'Z']
        
    
    def load(self):
        if self.type == 'train':
            return self.load_train()
        elif self.type == 'test':
            return self.load_test()
        elif self.type == 'both':
            return self.load_both()
        else:
            raise ValueError('Invalid type. Possible options are train|test|both')
        
    def load_train(self):
        return torchvision.datasets.ImageFolder(os.path.join(self.root, 'train/'), transform=self.train_transform)  
    
    def load_test(self):
        return torchvision.datasets.ImageFolder(os.path.join(self.root, 'test/'), transform=self.test_transform)
    
    def load_both(self):
        return {'train': self.load_train(), 'test': self.load_test()}
    