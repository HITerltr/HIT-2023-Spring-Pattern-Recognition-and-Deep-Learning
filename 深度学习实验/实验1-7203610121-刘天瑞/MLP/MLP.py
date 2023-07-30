# -*- coding: gbk -*-
"""
Python 3.9.7
torch 1.12.1
torchvision 0.13.1
device: cuda 11.8
"""
from doctest import OutputChecker
from telnetlib import OUTMRK
from tkinter import OUTSIDE
import torch
import torch.nn as nn
import torch.nn.functional as Function
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 超参数
batch_size = 128
epoch = 50
device = torch.device('cuda')

# 加载MNIST手写体数字数据集
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
train_data = torchvision.datasets.MNIST(root = './Data', train = True, download = True, transform = transform)
test_data = torchvision.datasets.MNIST(root = './Data', train = False, download = True, transform = transform)
train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle = True)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()# 调用父类的构造函数
        layer_1 = 512
        layer_2 = 512
        self.fc_1 = nn.Linear(28 * 28, layer_1)
        self.fc_2 = nn.Linear(layer_1, layer_2)
        self.fc_3 = nn.Linear(layer_2, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, out):
        out = out.view(-1, 28 * 28)

        out = Function.relu(self.fc_1(out))
        out = self.dropout(out)

        out = Function.relu(self.fc_2(out))
        out = self.dropout(out)

        out = Function.log_softmax(self.fc_3(out), dim = 1)
        return out


def train(model):
    #  保存最佳精确度
    best_accuracy = 0
    # 保存损失和精确值（绘制折线图时用到）
    loss_list = []
    acc_list = []
    # 定义损失函数和优化器
    loss_f = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.001)
    # 开始训练模型
    for a in range(epoch):
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            # 部署数据到GPU上并且计算输出值
            data, target = data.to(device), target.to(device)
            output = model(data)
            # 计算损失值
            loss = loss_f(output, target)
            # 进行反向传播预测
            loss.backward()
            # 将参数更新到神经网络中
            optimizer.step()
            # 计算训练损失值
            train_loss += loss.item() * data.size(0)
        train_loss /= len(train_loader.dataset)
        print('Epoch Times:{}   Training Loss: {:.4f}'.format(a + 1, train_loss), end = '')
        loss_list.append(train_loss)
        # 每完成一个epoch的模型训练，都需要在测试集上测试准确率
        accuracy = test(model)
        acc_list.append(accuracy)
        print("   Accuracy:{:.4f}".format(accuracy))
        # 将最佳模型保存
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            path = "./Model/BEST_MLP.ckpt"
            torch.save(model.state_dict(), path)
    print("Training Finished!")
    # 绘制损失和准确度曲线
    fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize = (5, 7))
    ax_1.set_title('Loss Line Chart')
    ax_2.set_title('Accuracy Line Chart')
    ax_1.set_ylim([0, 0.5])
    ax_2.set_ylim([0.8, 1])
    ax_1.plot(loss_list, color = '#0000FF')
    ax_2.plot(acc_list, color = '#FF0000')
    plt.show()


def test(model):
    # 初始化参量，定义损失函数
    correct = 0.0
    test_loss = 0.0
    loss_f = torch.nn.CrossEntropyLoss().cuda()
    # 模型验证
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            # 部署数据到GPU上并且计算输出值
            data, target = data.to(device), target.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失
            test_loss += loss_f(output, target).item() * data.size(0)
            # 找到概率值最大的下标
            pred = output.argmax(dim = 1)
            # 累计正确率
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
    return accuracy


def load_model(path = "./Model/BEST_MLP.ckpt"):
    #  载入模型进行测试
    MLP_model = MLP()
    MLP_model.load_state_dict(torch.load(path))
    MLP_model = MLP_model.cuda()
    accuracy = test(MLP_model)
    print("Load Model | Accuracy:{:.4f}".format(accuracy))


def train_model():
    MLP_model = MLP()
    # 将模型放到GPU上
    MLP_model = MLP_model.cuda()
    train(MLP_model)


if __name__ == "__main__":
    #训练模型
    train_model()
    #载入模型
    load_model(path="./Model/BEST_MLP.ckpt")

