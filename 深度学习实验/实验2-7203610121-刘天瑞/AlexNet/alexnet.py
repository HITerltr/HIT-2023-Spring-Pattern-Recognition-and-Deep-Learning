# -*- coding: gbk -*-
import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes = 101, dropout = 0.5):
        super(AlexNet, self).__init__()# ���ø���Ĺ��캯��
        self.features = nn.Sequential(# Sequential����ṹ�������ϳ��µĲ�ṹ
            nn.Conv2d(3, 48, kernel_size = 11, stride = 4, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0),

            nn.Conv2d(48, 128, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0),

            nn.Conv2d(128, 192, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),

            nn.Conv2d(192, 192, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),

            nn.Conv2d(192, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p = dropout),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace = True),
            nn.Dropout(p = dropout),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace = True),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.classifier(x)
        return x
