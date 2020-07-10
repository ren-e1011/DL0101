import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import glob
from tqdm.notebook import tqdm
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pytorch_lightning as pl
import torch.nn.functional as F

NUM_CLASSES = 10
LOSS_FUNC = F.cross_entropy


# class ImageClassifier(nn.Module):
class DeepSet(pl.LightningModule):
    
    def __init__(self):        
        super().__init__()
        
        self.counter = 0
        
        self.hidsize = 32*32
        
        
        #MOD
        self.embedding = nn.Linear(2,self.hidsize)
        
        #LeNet
#         self.conv1 = nn.Conv2d(1, 6, (5,5), padding=2)
#         self.conv2 = nn.Conv2d(6, 16, (5,5))
#         self.fc1   = nn.Linear(16*5*5, 120)
        self.fc1  = nn.Linear(self.hidsize,120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        
        
    def forward(self, batch):
        
        batch.requires_grad_(True)
        
        out = self.embedding(batch)
        
        out = F.relu(out)
        
        out = torch.mean(out,dim=1)
        
        #LeNet
#         x = F.max_pool2d(F.relu(self.conv1(out)), (2,2))
#         x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
#         x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(out))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)


        return out
    
    
    def train_dataloader(self):
        dataset_train = CustomDataset(path_to_training_data)
        batch_sampler = CustomBatchSampler(dataset_train.n_points, batch_size=50)
        return DataLoader(dataset_train,batch_sampler=batch_sampler, num_workers=4)

    def val_dataloader(self):
        dataset_val = CustomDataset(path_to_validation_data)
        batch_sampler_val = CustomBatchSampler(dataset_val.n_points,batch_size=50)
        # MOD num_workers on local
        return DataLoader(dataset_val,batch_sampler=batch_sampler_val,num_workers=4)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    
    def training_step(self,batch,batch_idx):
        minib,target = batch
        yhat = self(minib)
        loss = LOSS_FUNC(yhat,target)
#         # add logging
#         logs = {'train_loss': loss,'train_accuracy':accuracy}
        logs = {'train_loss':loss}
#         print('logs',logs)
        return {'loss': loss, 'log': logs}
#         return loss
    
    def validation_step(self,batch,batch_idx):
        minib, target = batch
        yhat = self(minib)
        # mod to torch.tensor
        loss = LOSS_FUNC(yhat,target)
        
        pred = yhat.argmax(dim=1, keepdim=True) #get ix of max log-proba
        correct = pred.eq(target.view_as(pred)).sum().item()
        #mod to torch.tensor forr avg_acc
        accuracy = torch.tensor(correct/len(minib))
        
        step_dict = {'val_step_loss':loss, 'val_step_acc':accuracy}
        return step_dict
        
    
    def validation_epoch_end(self, outputs):
        #TODO imbalanced given each batch is of a different size...
        avg_loss = torch.stack([x['val_step_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_step_acc'] for x in outputs]).mean()

        results = {
            'log': {'val_acc': avg_acc,
                    'val_loss': avg_loss
                   }
        }
        print ('epoch',self.counter,':',results) 
        self.counter +=1
        return results