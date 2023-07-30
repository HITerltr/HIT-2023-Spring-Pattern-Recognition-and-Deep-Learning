# -*- coding: gbk -*-
import argparse
import torch
import torch.cuda
from alexnet import AlexNet
from caltech101 import Caltech101
from torchvision import transforms
from load_img_data import load_img_data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 构建输入参数
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices = ['train', 'test'], default = 'test')
args = parser.parse_args()

# 超参数设置
batch_size = 64
epoch = 80

# 设定设备选项
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化tensorboard writer
tb_writer = SummaryWriter(log_dir = "./Runs")
tags = ["Train_loss", "Accuracy", "Learning rate"]


def train(train_loader, test_loader, val_loader, model, path = "./Model/Best_AlexNet.ckpt"):
    # 保存最佳accuracy
    best_accuracy = 0
    # 保存loss和accuracy值（绘制曲线图时会用到）
    # loss_list = []
    # acc_list = []
    # 定义损失函数和优化器
    loss_func = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.0003)# 学习率设置为0.0003
    # 开始训练
    for i in range(epoch):
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            # 部署数据至GPU并计算输出
            data, target = data.to(device), target.to(device)
            output = model(data)
            # 计算损失
            loss = loss_func(output, target)
            # 反向传播
            loss.backward()
            # 将参数更新至网络中
            optimizer.step()
            # 计算损失
            train_loss += loss.item() * data.size(0)
        mean_loss = train_loss / len(train_loader.dataset)
        print('Epoch:{}   Train Loss: {:.4f}'.format(i + 1, mean_loss), end='')
        # 每完成一轮epoch的训练，都需要在测试集上测试准确率
        accuracy = test(val_loader, model)
        print("   accuracy:{:.4f}".format(accuracy))
        # 保存最佳模型
        if accuracy >best_accuracy:
           best_accuracy = accuracy
           torch.save(model.state_dict(), path)
        # 将相关参数写入tensorboard图形显示工具
        tb_writer.add_scalar(tags[0], mean_loss, i)
        tb_writer.add_scalar(tags[1], accuracy, i)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], i)
    print("Finished")
    # 在测试集上验证结果
    test_accuracy = test(test_loader, model)
    print("Test Set accuracy:{:.4f}".format(test_accuracy))


def test(test_loader, model):
    # 初始化参量，定义损失函数
    correct = 0.0
    test_loss = 0.0
    loss_f = torch.nn.CrossEntropyLoss().cuda()
    # 验证模型
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            # 部署数据
            data, target = data.to(device), target.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失值
            test_loss += loss_f(output, target).item() * data.size(0)
            # 找到概率值最大的下标
            pred = output.argmax(dim = 1)
            # 累计正确率
            correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
    return accuracy


def load_model(test_loader, model, path = "./Model/Best_AlexNet.ckpt"):
    """载入模型进行测试"""
    model.load_state_dict(torch.load(path))
    model = model.cuda()
    accuracy = test(test_loader, model)
    print("Load model | Accuracy:{:.4f}".format(accuracy))


def train_model(train_loader, testloader, val_loader, model):
    """将模型放到GPU的cuda上"""
    model = model.cuda()
    train(train_loader, testloader, val_loader, model)


if __name__ == '__main__':
    train_path, train_label, test_path, test_label, val_path, val_label = load_img_data()

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),  # 将给定的图像随机裁剪为不同的大小和高宽比，然后缩放所裁剪得到的图像为指定大小
            transforms.RandomHorizontalFlip(),  # 以给定的概率（默认为0.5）水平（随机）翻转图像
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "test": transforms.Compose([
            transforms.RandomResizedCrop(224),  # 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为指定大小
            transforms.RandomHorizontalFlip(),  # 以给定的概率（默认为0.5）水平（随机）翻转图像
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    train_dataset = Caltech101(img_path = train_path, img_class = train_label, transform = data_transform["train"])
    test_dataset = Caltech101(img_path = test_path, img_class = test_label, transform = data_transform["test"])
    val_dataset = Caltech101(img_path = val_path, img_class = val_label, transform = data_transform["val"])

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

    # 构建卷积神经网络
    model = AlexNet(num_classes = 101, dropout = 0.5)
    model.to(device)
    # 将网络模型结构写入tensorboard图形显示工具
    init_img = torch.zeros((1, 3, 224, 224), device = device)
    tb_writer.add_graph(model, init_img)

    if args.mode == 'train':
        train_model(train_loader, test_loader, val_loader, model)
    elif args.mode == 'test':
        load_model(test_loader, model, path = "./Model/Best_AlexNet.ckpt")
