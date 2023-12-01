import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def __init__(self, beta=1.0, inplace=False):
        super(Swish, self).__init__()
        self.beta = torch.nn.Parameter(torch.tensor(data=beta, dtype=torch.float32), requires_grad=True)
        self.inplace = inplace
    def forward(self, x):
        return x * torch.sigmoid(x * self.beta)

class MyCNN(nn.Module):
    """This class implements the CNN architecture in PyTorch"""

    def __init__(self, n_classes,  dropout_conv=0.0, dropout_fc=0.0,
                 act = nn.ReLU,
                #  act = nn.LeakyReLU,
                #  act = nn.SiLU,
                #  act = Swish,
                 ):
        """
        Constructor for the MyCNN class.
        """
        super(MyCNN, self).__init__()
        max_pool_kernel_size, max_pool_kernel_stride = 2, None
        self.conv = nn.Sequential(
                    # 1
                    nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=0, bias=True, padding_mode='zeros'),
                    nn.BatchNorm2d(num_features=10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    act(),
                    nn.MaxPool2d(kernel_size = max_pool_kernel_size, stride = max_pool_kernel_stride),
                    # 2
                    nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=0, bias=True, padding_mode='zeros'),
                    nn.BatchNorm2d(num_features=20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    act(),
                    nn.MaxPool2d(kernel_size = max_pool_kernel_size, stride = max_pool_kernel_stride),
                    # 3
                    nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3, stride=1, padding=0, bias=True, padding_mode='zeros'),
                    nn.BatchNorm2d(num_features=30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    act(),
                    nn.Dropout2d(p=dropout_conv),
              )

        self.fc = nn.Sequential(
                      nn.Linear(in_features=30 * 3 * 3, out_features=270, bias=True),
                      act(),
                      nn.Dropout(p=dropout_fc),
                      nn.Linear(in_features=270, out_features=n_classes, bias=True),
                )
        self.softmax = nn.LogSoftmax(dim=1)

        # # Initialize the weights of each trainable layer of your network using xavier_uniform initialization
        # for m in self.conv :
        #     if isinstance(m, nn.Conv2d): torch.nn.init.xavier_uniform_(m.weight)
        # for m in self.fc :
        #       if isinstance(m, nn.Linear): torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        h = self.conv(x)
        h = torch.flatten(h, start_dim=1, end_dim=-1)
        h = self.fc(h)
        h = self.softmax(h)
        return h

    def predict(self, x):
        return self.forward(x).argmax(dim=-1)


class MyCNN_2(nn.Module):
    """This class implements the CNN architecture in PyTorch"""

    def __init__(self, n_classes,
                 dropout_conv=0.0, dropout_fc=0.0,
                 act = nn.ReLU,
                #  act = nn.LeakyReLU,
                #  act = nn.SiLU,
                #  act = Swish,
                 ):
        """
          Constructor for the MyCNN class.
        """
        super(MyCNN_2, self).__init__()
        max_pool_kernel_size, max_pool_kernel_stride = 2, None
        padding_mode='zeros'
        padding=1
        dropout = dropout_conv
        self.conv = nn.Sequential(
                    # 1
                    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=padding, bias=True, padding_mode=padding_mode),
                    nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    act(),
                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=padding, bias=True, padding_mode=padding_mode),
                    nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    act(),
                    nn.MaxPool2d(kernel_size = max_pool_kernel_size, stride = max_pool_kernel_stride),
                    nn.Dropout2d(p=dropout),
                    # 2
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=padding, bias=True, padding_mode=padding_mode),
                    nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    act(),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=padding, bias=True, padding_mode=padding_mode),
                    nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    act(),
                    nn.MaxPool2d(kernel_size = max_pool_kernel_size, stride = max_pool_kernel_stride),
                    nn.Dropout2d(p=dropout),
                    # 3
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=padding, bias=True, padding_mode=padding_mode),
                    nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    act(),
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=padding, bias=True, padding_mode=padding_mode),
                    nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    act(),
                    nn.MaxPool2d(kernel_size = max_pool_kernel_size, stride = max_pool_kernel_stride),
                    nn.Dropout2d(p=dropout),
                    #
                    nn.Flatten(),
              )

        dropout = dropout_fc
        # self.fc = nn.Sequential(
        #               nn.Linear(in_features=128*3*3, out_features=512, bias=True),
        #               act(),
        #               nn.BatchNorm1d(num_features=512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #               nn.Dropout(p=dropout),
        #               #
        #               nn.Linear(in_features=512, out_features=256, bias=True),
        #               act(),
        #               nn.BatchNorm1d(num_features=256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #               nn.Dropout(p=dropout),
        #               #
        #               nn.Linear(in_features=256, out_features=64, bias=True),
        #               act(),
        #               nn.BatchNorm1d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #               nn.Dropout(p=dropout),
        #               #
        #               nn.Linear(in_features=64, out_features=self.n_classes, bias=True),
        #         )
        self.fc = nn.Sequential(
                      nn.Linear(in_features=128*3*3, out_features=512, bias=True),
                      act(),
                      nn.Dropout(p=dropout),
                      nn.Linear(in_features=512, out_features=n_classes, bias=True),
                )
        self.softmax = nn.LogSoftmax(dim=1)

        # # Initialize the weights of each trainable layer of your network using xavier_uniform initialization
        # for m in self.conv :
        #     if isinstance(m, nn.Conv2d): torch.nn.init.xavier_uniform_(m.weight)
        # for m in self.fc :
        #       if isinstance(m, nn.Linear): torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        h = self.conv(x)
        #h = torch.flatten(h, start_dim=1, end_dim=-1)
        h = self.fc(h)
        h = self.softmax(h)
        return h

    def predict(self, x):
        return self.forward(x).argmax(dim=-1)


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet(nn.Module):
    def __init__(self, n_classes,
                 dropout_conv=0.0, dropout_fc=0.0,
                 act = nn.ReLU,
                #  act = nn.LeakyReLU,
                #  act = nn.SiLU,
                #  act = Swish,
                 ):
        super(ResNet, self).__init__()

                                             # 1 x 28 x 28
        self.conv1 = conv_block(1, 64)       # 64 x 28 x 28 : 28+2*1-3+1=28
        self.conv2 = conv_block(64, 128, pool=True)    # 128 x 16 x 16
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128)) # 128 x 16 x 16
        self.drop1 = nn.Dropout2d(p=dropout_conv)

        self.conv3 = conv_block(128, 256, pool=True)   # 256 x 8 x 8
        self.conv4 = conv_block(256, 512, pool=True)   # 512 x 4 x 4
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512)) # 512 x 4 x 4
        self.drop2 = nn.Dropout2d(p=dropout_conv)

        self.fc = nn.Sequential(nn.MaxPool2d(2), # 512 x 1 x 1
                                        nn.Flatten(),    # 512
                                        nn.Linear(512, 256),
                                        act(),
                                        nn.Dropout(p=dropout_fc),
                                        nn.Linear(256, n_classes),
                                        ) # 24

        self.softmax = nn.LogSoftmax(dim=1)

        # # Initialize the weights of each trainable layer of your network using xavier_uniform initialization
        # for m in self.conv :
        #     if isinstance(m, nn.Conv2d): torch.nn.init.xavier_uniform_(m.weight)
        # for m in self.fc :
        #       if isinstance(m, nn.Linear): torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.res1(h) + h
        h = self.drop1(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.res2(h) + h
        h = self.drop2(h)
        h = self.fc(h)
        h = self.softmax(h)
        return h

    def predict(self, x):
        return self.forward(x).argmax(dim=-1)


class ResNet9(ResNet):
    def __init__(self, n_classes,
                 dropout_conv=0.0, dropout_fc=0.0,
                 act = nn.ReLU,
                #  act = nn.LeakyReLU,
                #  act = nn.SiLU,
                #  act = Swish,
                 ):
        super(ResNet9, self).__init__(n_classes, dropout_conv, dropout_fc, act)

class Net(nn.Module):
    """
    Internal ensembles of the model,
    https://jimut123.github.io/blogs/MNIST_rank_17.html
    """
    def __init__(self, n_classes):
        super(Net, self).__init__()

        # define a conv layer with output channels as 16, kernel size of 3 and stride of 1
        self.conv11 = nn.Conv2d(1, 16, 3, 1) # Input = 1x28x28  Output = 16x26x26
        self.conv12 = nn.Conv2d(1, 16, 5, 1) # Input = 1x28x28  Output = 16x24x24
        self.conv13 = nn.Conv2d(1, 16, 7, 1) # Input = 1x28x28  Output = 16x22x22
        self.conv14 = nn.Conv2d(1, 16, 9, 1) # Input = 1x28x28  Output = 16x20x20

        # define a conv layer with output channels as 32, kernel size of 3 and stride of 1
        self.conv21 = nn.Conv2d(16, 32, 3, 1) # Input = 16x26x26 Output = 32x24x24
        self.conv22 = nn.Conv2d(16, 32, 5, 1) # Input = 16x24x24 Output = 32x20x20
        self.conv23 = nn.Conv2d(16, 32, 7, 1) # Input = 16x22x22 Output = 32x16x16
        self.conv24 = nn.Conv2d(16, 32, 9, 1) # Input = 16x20x20  Output = 32x12x12

        # define a conv layer with output channels as 64, kernel size of 3 and stride of 1
        self.conv31 = nn.Conv2d(32, 64, 3, 1) # Input = 32x24x24 Output = 64x22x22
        self.conv32 = nn.Conv2d(32, 64, 5, 1) # Input = 32x20x20 Output = 64x16x16
        self.conv33 = nn.Conv2d(32, 64, 7, 1) # Input = 32x16x16 Output = 64x10x10
        self.conv34 = nn.Conv2d(32, 64, 9, 1) # Input = 32x12x12 Output = 64x4x4


        # define a max pooling layer with kernel size 2
        self.maxpool = nn.MaxPool2d(2) # Output = 64x11x11
        #self.maxpool1 = nn.MaxPool2d(1)
        # define dropout layer with a probability of 0.25
        self.dropout1 = nn.Dropout(0.25)
        # define dropout layer with a probability of 0.5
        self.dropout2 = nn.Dropout(0.5)

        # define a linear(dense) layer with 128 output features
        self.fc11 = nn.Linear(64*11*11, 256)
        self.fc12 = nn.Linear(64*8*8, 256)      # after maxpooling 2x2
        self.fc13 = nn.Linear(64*5*5, 256)
        self.fc14 = nn.Linear(64*2*2, 256)

        # define a linear(dense) layer with output features corresponding to the number of classes in the dataset
        self.fc21 = nn.Linear(256, 128)
        self.fc22 = nn.Linear(256, 128)
        self.fc23 = nn.Linear(256, 128)
        self.fc24 = nn.Linear(256, 128)

        self.fc33 = nn.Linear(128*4,n_classes)

    def forward(self, inp):
        # Use the layers defined above in a sequential way (folow the same as the layer definitions above) and 
        # write the forward pass, after each of conv1, conv2, conv3 and fc1 use a relu activation. 


        x = F.relu(self.conv11(inp))
        x = F.relu(self.conv21(x))
        x = F.relu(self.maxpool(self.conv31(x)))
        #print(x.shape)
        #x = torch.flatten(x, 1)
        x = x.view(-1,64*11*11)
        x = self.dropout1(x)
        x = F.relu(self.fc11(x))
        x = self.dropout2(x)
        x = self.fc21(x)

        y = F.relu(self.conv12(inp))
        y = F.relu(self.conv22(y))
        y = F.relu(self.maxpool(self.conv32(y)))
        #x = torch.flatten(x, 1)
        y = y.view(-1,64*8*8)
        y = self.dropout1(y)
        y = F.relu(self.fc12(y))
        y = self.dropout2(y)
        y = self.fc22(y)

        z = F.relu(self.conv13(inp))
        z = F.relu(self.conv23(z))
        z = F.relu(self.maxpool(self.conv33(z)))
        #x = torch.flatten(x, 1)
        z = z.view(-1,64*5*5)
        z = self.dropout1(z)
        z = F.relu(self.fc13(z))
        z = self.dropout2(z)
        z = self.fc23(z)

        ze = F.relu(self.conv14(inp))
        ze = F.relu(self.conv24(ze))
        ze = F.relu(self.maxpool(self.conv34(ze)))
        #x = torch.flatten(x, 1)
        ze = ze.view(-1,64*2*2)
        ze = self.dropout1(ze)
        ze = F.relu(self.fc14(ze))
        ze = self.dropout2(ze)
        ze = self.fc24(ze)

        out_f = torch.cat((x, y, z, ze), dim=1)
        #out_f1 = torch.cat((out_f, ze), dim=1)
        out = self.fc33(out_f)

        output = F.log_softmax(out, dim=1)
        return output

    def predict(self, x):
        return self.forward(x).argmax(dim=-1)