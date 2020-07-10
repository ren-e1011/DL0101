import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from torch.utils.data import Dataset, DataLoader
import glob
import json
from tqdm.notebook import tqdm
import torch.optim as optim
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

#this function is the edge update function - 

class EdgeNetwork(nn.Module):
    def __init__(self, nodemb, edgemb, nodefeat=2):
        super(EdgeNetwork, self).__init__()
        
        # constants
        self.nodemb = nodemb
        self.nodefeat = nodefeat
        self.edgemb = edgemb
           
        self.edgseq = nn.Sequential(

            
            #noderepsize is curr equal nodemb
            nn.Linear(self.nodemb*2 + self.nodefeat*2 + 1, 128),
            
            nn.ReLU(True),
            
            nn.Linear(128, self.edgemb),
            
            nn.ReLU(True)
        )

 # EdgeBatch x       
    def forward(self, x):

        #your input x is an object with the following properties:
        #x.dst['node_features'], x.dst['node_hidden_state']
        #x.src['node_features'], x.src['node_hidden_state']
        #x.data['distance']

    
        data = x.data['distance'].unsqueeze(-1)
                      
        out = torch.cat([x.dst['node_features'],x.dst['node_hidden_state'], x.src['node_features'],x.src['node_hidden_state'],data],dim=1)

        out = self.edgseq(out)

        return {'edge_hidden_representation': out }

class NodeNetwork(nn.Module):
    def __init__(self,nodemb, edgemb, nodefeat=2):
        super(NodeNetwork, self).__init__()
        
        self.edgemb = edgemb
        self.nodefeat = nodefeat
        self.nodemb = nodemb       
        self.nodeseq = nn.Sequential(

            #noderepsize is curr equal nodemb
            nn.Linear(self.nodemb + self.nodefeat+ self.edgemb,128),
            
            nn.ReLU(True),

            nn.Linear(128, self.nodemb),

            nn.ReLU(True)
        )
  
    def forward(self, x):

#         edge_hid_rep_flat = torch.mean(x.mailbox['edge_hidden_representation'],dim=1)
#         edge_hid_rep_flat = edge_hid_rep_flat.squeeze(1)
        
        edge_hid_rep_flat = torch.sum(x.mailbox['edge_hidden_representation'],dim=1,keepdim=False)

        
        out = torch.cat([x.data['node_hidden_state'],x.data['node_features'],edge_hid_rep_flat],dim=1)
        
        out = self.nodeseq(out)
        
#         #- and then apply some fully connected neural network
        # first epoch
#         if out.size()[1] == self.noderep_size + self.nodefeat + self.edgemb:
#             out = F.relu(self.fc1a(out))
            
#         else:
#             out = F.relu(self.fc1b(out))
            
#         out = F.relu(self.fc1(out))
#         out = F.relu(self.fc2(out))
        
        # 1. sum on mailbox
        # 2. concat with node_hidden_state, node_features
        # return a new hidden state for the node, updates hidden states of nodes -- make same size? :/
        return {'node_hidden_state': out }

class EdgeClassifier(nn.Module):
    def __init__(self, nodemb, nodefeat=2):
        super(EdgeClassifier, self).__init__()
        
        self.nodemb = nodemb
        self.nodefeat = nodefeat
        
        self.classeq = nn.Sequential(
        
            nn.Linear(self.nodemb*2 + self.nodefeat*2  +1,self.nodemb),
            nn.ReLU(True),
            nn.Linear(self.nodemb,1),
            # TRY
            nn.Sigmoid()
#             nn.ReLU(True)
        )

   

     
    def forward(self, x):

    
        data = x.data['distance'].unsqueeze(-1)
              
#         print('data.size()',data.size())
        
        out = torch.cat([x.dst['node_features'],x.dst['node_hidden_state'], x.src['node_features'],x.src['node_hidden_state'],data],dim=1)


        out = self.classeq(out)
        
        #put them together with torch.cat
        
        #use a neural network to create an edge hidden represetation - (select size)
        
        #you return a dictionary with what you want to "send" to the reciving node
        
        #sent to nodes

        return {'edge_class_prediction': out }
#         return {'edge_class_prediction': output }

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        # you need to create a network that 
        # will initialize your node hidden state based only on the node features -
#         self.node_representation_size = 32 
        # make everything equal for a minute
#         self.node_representation_size = 64
#         self.edge_representation_size = 128 
        # fc net, linear layers
    
    
        
        self.nodemb = 64 #output of final layer in NodeNetwork
        self.edgemb = 64 #output of final layer in EdgeNetwork  
        
        self.node_init = nn.Sequential(

            nn.Linear(2,128),

            nn.ReLU(True),
            
            nn.Linear(128,self.nodemb),

            nn.ReLU(True)
        )
        
        
#         self.edge_network = EdgeNetwork(self.node_representation_size, self.edgemb)
        self.edge_network = EdgeNetwork(self.nodemb, self.edgemb)
        
        self.node_network = NodeNetwork(self.nodemb, self.edgemb)
        
        self.edge_classifier = EdgeClassifier(self.nodemb)
        
        #this edge classifier is also an edge update function - 
        #but it needs to return something of size 1 (the edge class prediction)
        #so either create a different model for this, or make the EdgeNetwork configurable
        
        # return one number per edge. new module or reconfig EdgeNetwork __init__(self,outputsize,outputname)
        
    def forward(self, g):
        
        # init node_hidden_state to step thru node_init net and populate node_features
        g.ndata['node_hidden_state'] = self.node_init(g.ndata['node_features'])
        
        number_of_iterations = 16
        
        for i in range(number_of_iterations):

            g.update_all(self.edge_network,self.node_network)
            
            # NodeNetwork returns a new hidden state for the node, updates hidden states of nodes of size nodemb -- post-iteration 1
#             self.node_representation_size = self.nodemb
            
            
        # Edge classification task
        # Edge network, return one number per edge

        g.apply_edges(self.edge_classifier)
        
        #and extract its output, should size-match input (from collate()) 
        # output same number of edges as dgl graph
        out = g.edata['edge_class_prediction']

        
        return out 