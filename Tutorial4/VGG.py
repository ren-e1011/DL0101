import pytorch_lightning as pl
from torchvision import datasets, transforms
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



class VGG(pl.LightningModule):

    def __init__(self, num_classes=NUM_CLASSES):
#         super(VGG, self).__init__()
        super().__init__()
        # Added to .py
        vgg19 = models.vgg19(pretrained=True)
        
        self.features = vgg19.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

            
    def forward(self, x):
        out = self.features(x)
        # what does this accomplish 
        out = self.avgpool(out)
        out = torch.flatten(out,1)
        out = self.classifier(out)
                
        return out
    
    def train_dataloader(self):
        dataset_train = CustomDataset(path_to_training_data)
# seemingly CustomDataset handles transformations, tensor...collate_fn=_collate unnecessary                
        return DataLoader(dataset_train,batch_size=BATCH_SIZE,shuffle=True)

    def val_dataloader(self):
        dataset_val = CustomDataset(path_to_validation_data)
        return DataLoader(dataset_val,batch_size=BATCH_SIZE,shuffle=False)
    
    def configure_optimizers(self):
    
        return torch.optim.Adam(self.parameters(),lr=0.0001)

        
    
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
        
        pred = out.argmax(dim=1, keepdim=True) #get ix of max log-proba
        correct = pred.eq(target.view_as(pred)).sum().item()
        
        return {'val_loss': loss, 'correct': correct}
        
    
    def validation_epoch_end(self,outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss':avg_loss}
        return {'avg_val_loss':avg_loss, 'log':tensorboard_logs}

