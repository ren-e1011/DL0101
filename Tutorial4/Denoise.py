import pytorch_lightning as pl
from torchvision import datasets, transforms
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# https://web.stanford.edu/class/cs331b/2016/projects/zhao.pdf
# https://github.com/pgtgrly/Convolution-Deconvolution-Network-Pytorch/blob/master/Neural_Network_Class.py
# https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-deconv.ipynb
class Denoise(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.deconv5 = nn.ConvTranspose2d(512, 256, 3, padding=1)
        self.deconv4 = nn.ConvTranspose2d(256, 128, 3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128,64, 3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 5, padding=2)
        self.deconv1 = nn.ConvTranspose2d(32, 1, 5, padding=2)
        
        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out1,ix1 = self.pool1(F.relu(self.conv1(x)))
        out2,ix2 = self.pool2(F.relu(self.conv2(out1)))
        out3,ix3 = self.pool3(F.relu(self.conv3(out2)))
        out4,ix4 = self.pool4(F.relu(self.conv4(out3)))
        out5,ix5 = self.pool5(F.relu(self.conv5(out4)))
        
        out6 = F.relu(self.deconv5(self.unpool5(out5, ix5)))
        out7 = F.relu(self.deconv4(self.unpool4(out4, ix4)))
        out8 = F.relu(self.deconv3(self.unpool3(out3, ix3)))
        out9 = F.relu(self.deconv2(self.unpool2(out2, ix2)))
        out = F.relu(self.deconv1(self.unpool1(out1, ix1)))
        
        return out
    
    def train_dataloader(self):
        dataset_train = CustomDataset('/storage/cats.npy',0,800)
# seemingly CustomDataset handles transformations, tensor...collate_fn=_collate unnecessary                
        return DataLoader(dataset_train,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)

    def val_dataloader(self):
        dataset_val = CustomDataset('/storage/cats.npy',800,1000)
        return DataLoader(dataset_val,batch_size=BATCH_SIZE,shuffle=False,num_workers=4)
    
    def configure_optimizers(self):
        optimizer1 = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler1 = StepLR(optimizer1, step_size=1)
        return [optimizer1], [scheduler1]
        
    
    def training_step(self,batch,batch_idx):
        minib,target = batch
        out = self.forward(minib)
        loss = LOSS_FUNC(out,target)
        # add logging
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}
#         return loss
    
    def validation_step(self,batch,batch_idx):
        minib, target = batch
        out = self.forward(minib)
        loss = LOSS_FUNC(out,target)
        return {'val_loss': loss}
#         return {'val_loss': loss, 'correct': correct}
        
    
    def validation_epoch_end(self,outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss':avg_loss}
        return {'avg_val_loss':avg_loss, 'log':tensorboard_logs}
        