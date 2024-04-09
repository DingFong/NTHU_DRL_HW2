import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Q_net(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Q_net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_output_shape = self.get_conv_output_size(input_shape) # 3136
        
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def get_conv_output_size(self, input_shape):
        input_ = torch.zeros((1, *input_shape))
        # print(np.prod(self.conv(input_).size()))
        return np.prod(self.conv(input_).size())
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)

        return self.fc(conv_out)
    

# class Q_net(nn.Module):
#     def __init__(self, input_shape, n_actions):
#         super(Q_net, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(input_shape[0], 32, kernel_size=4, stride = 2),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, kernel_size=4, stride = 2),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU(),
#         )
#         conv_output_shape = self.get_conv_output_size(input_shape)

#         self.fc = nn.Sequential(
#             nn.Linear(conv_output_shape, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, n_actions)
#         )

#     def get_conv_output_size(self, input_shape):
#         input_ = torch.zeros((1, *input_shape))
#         # print(np.prod(self.conv(input_).size()))
#         return np.prod(self.conv(input_).size())
    
#     def forward(self, x):
#         conv_out = self.conv(x).view(x.size()[0], -1)

#         return self.fc(conv_out)