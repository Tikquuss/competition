import torch
import torch.nn as nn

class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = torch.nn.Parameter(torch.tensor(data=beta, dtype=torch.float32), requires_grad=True)
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

class ResNet9(nn.Module):
    def __init__(self, n_classes,
                 dropout_conv=0.0, dropout_fc=0.0,
                 act = nn.ReLU,
                #  act = nn.LeakyReLU,
                #  act = nn.SiLU,
                #  act = Swish,
                 ):
        super(ResNet9, self).__init__()

                                             # 3 x 28 x 28
        self.conv1 = conv_block(1, 64)       # 64 x 28 x 28 : 28+2*1-3+1=28
        self.conv2 = conv_block(64, 128, pool=True)    # 128 x 16 x 16
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128)) # 128 x 16 x 16
        self.drop1 = nn.Dropout2d(p=dropout_conv)

        self.conv3 = conv_block(128, 256, pool=True)   # 256 x 8 x 8
        self.conv4 = conv_block(256, 512, pool=True)   # 512 x 4 x 4
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512)) # 512 x 4 x 4
        self.drop2 = nn.Dropout2d(p=dropout_conv)

        # self.classifier = nn.Sequential(nn.MaxPool2d(2), # 512 x 1 x 1
        #                                 nn.Flatten(),    # 512
        #                                 nn.Linear(512, n_classes),
        #                                 nn.Dropout(p=dropout_fc),
        #                                 ) # 24

        self.fc = nn.Sequential(nn.MaxPool2d(2), # 512 x 1 x 1
                                        nn.Flatten(),    # 512
                                        nn.Linear(512, 256),
                                        act(),
                                        nn.Dropout(p=dropout_fc),
                                        nn.Linear(256, n_classes),
                                        ) # 24

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.drop1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.drop2(out)
        out = self.fc(out)
        out = self.softmax(out)
        return out

    def predict(self, x):
        return self.forward(x).argmax(dim=-1)