
# built on https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html#learn-more class Net(nn.Module):

class MnistConvNet(nn.Module):
    def __init__(self):
        super(MnistConvNet, self).__init__()

        # First 2D convolutional layer, taking in 1 input channel (image),
        # outputting 32 convolutional features, with a square kernel size of 3
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # Second 2D convolutional layer, taking in the 32 input layers,
        # outputting 64 convolutional features, with a square kernel size of 3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # Designed to ensure that adjacent pixels are either all 0s or all active
        # with an input probability
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        # First fully connected layer
        self.fc1 = nn.Linear(64*32*32, 128)
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(128, classes)

        # x represents our data
    def forward(self, x):
#         print('raw_x',x.size())
        # Pass data through conv1
        x = self.conv1(x)
#         print('nn.Conv2d (1,32,2,1) -->', x.size())
        # Use the rectified-linear activation function over x
        x = F.relu(x)
#         print('F.relu -->', x.size())
        x = self.conv2(x)
#         print('nn.Conv2d(32,64,3,1) -->', x.size())
        x = F.relu(x)
#         print('F.relu -->',x.size())

        # Run max pooling over x
        x = F.max_pool2d(x, 2)
#         print('F.max_pool2d(x,2) --> ',x.size())
        # Pass data through dropout1
        x = self.dropout1(x)
#         print('nn.Dropout2d(0.25) --> ',x.size())
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
#         print('torch.flatten(x,1) -->',x.size())
        # Pass data through fc1
        x = self.fc1(x)
#         print('nn.Linear(69*69,128) --> ',x.size())
        x = F.relu(x)
#         print('F.relu(x) -->',x.size())
        x = self.dropout2(x)
#         print('nn.Dropout2d(0.5) -->',x.size())
        x = self.fc2(x)
#         print('nn.Linear(128,classes) -->',x.size())

        # Apply softmax to x
        #       output = F.log_softmax(x, dim=1)
        output = x
        return output
    
