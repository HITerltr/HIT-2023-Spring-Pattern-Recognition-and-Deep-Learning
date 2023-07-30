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

# �����������
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices = ['train', 'test'], default = 'test')
args = parser.parse_args()

# ����������
batch_size = 64
epoch = 80

# �趨�豸ѡ��
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ʵ����tensorboard writer
tb_writer = SummaryWriter(log_dir = "./Runs")
tags = ["Train_loss", "Accuracy", "Learning rate"]


def train(train_loader, test_loader, val_loader, model, path = "./Model/Best_AlexNet.ckpt"):
    # �������accuracy
    best_accuracy = 0
    # ����loss��accuracyֵ����������ͼʱ���õ���
    # loss_list = []
    # acc_list = []
    # ������ʧ�������Ż���
    loss_func = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.0003)# ѧϰ������Ϊ0.0003
    # ��ʼѵ��
    for i in range(epoch):
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            # ����������GPU���������
            data, target = data.to(device), target.to(device)
            output = model(data)
            # ������ʧ
            loss = loss_func(output, target)
            # ���򴫲�
            loss.backward()
            # ������������������
            optimizer.step()
            # ������ʧ
            train_loss += loss.item() * data.size(0)
        mean_loss = train_loss / len(train_loader.dataset)
        print('Epoch:{}   Train Loss: {:.4f}'.format(i + 1, mean_loss), end='')
        # ÿ���һ��epoch��ѵ��������Ҫ�ڲ��Լ��ϲ���׼ȷ��
        accuracy = test(val_loader, model)
        print("   accuracy:{:.4f}".format(accuracy))
        # �������ģ��
        if accuracy >best_accuracy:
           best_accuracy = accuracy
           torch.save(model.state_dict(), path)
        # ����ز���д��tensorboardͼ����ʾ����
        tb_writer.add_scalar(tags[0], mean_loss, i)
        tb_writer.add_scalar(tags[1], accuracy, i)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], i)
    print("Finished")
    # �ڲ��Լ�����֤���
    test_accuracy = test(test_loader, model)
    print("Test Set accuracy:{:.4f}".format(test_accuracy))


def test(test_loader, model):
    # ��ʼ��������������ʧ����
    correct = 0.0
    test_loss = 0.0
    loss_f = torch.nn.CrossEntropyLoss().cuda()
    # ��֤ģ��
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            # ��������
            data, target = data.to(device), target.to(device)
            # ��������
            output = model(data)
            # ���������ʧֵ
            test_loss += loss_f(output, target).item() * data.size(0)
            # �ҵ�����ֵ�����±�
            pred = output.argmax(dim = 1)
            # �ۼ���ȷ��
            correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
    return accuracy


def load_model(test_loader, model, path = "./Model/Best_AlexNet.ckpt"):
    """����ģ�ͽ��в���"""
    model.load_state_dict(torch.load(path))
    model = model.cuda()
    accuracy = test(test_loader, model)
    print("Load model | Accuracy:{:.4f}".format(accuracy))


def train_model(train_loader, testloader, val_loader, model):
    """��ģ�ͷŵ�GPU��cuda��"""
    model = model.cuda()
    train(train_loader, testloader, val_loader, model)


if __name__ == '__main__':
    train_path, train_label, test_path, test_label, val_path, val_label = load_img_data()

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),  # ��������ͼ������ü�Ϊ��ͬ�Ĵ�С�͸߿�ȣ�Ȼ���������ü��õ���ͼ��Ϊָ����С
            transforms.RandomHorizontalFlip(),  # �Ը����ĸ��ʣ�Ĭ��Ϊ0.5��ˮƽ���������תͼ��
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "test": transforms.Compose([
            transforms.RandomResizedCrop(224),  # ������ͼ������ü�Ϊ��ͬ�Ĵ�С�Ϳ�߱ȣ�Ȼ���������ü��õ���ͼ��Ϊָ����С
            transforms.RandomHorizontalFlip(),  # �Ը����ĸ��ʣ�Ĭ��Ϊ0.5��ˮƽ���������תͼ��
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

    # �������������
    model = AlexNet(num_classes = 101, dropout = 0.5)
    model.to(device)
    # ������ģ�ͽṹд��tensorboardͼ����ʾ����
    init_img = torch.zeros((1, 3, 224, 224), device = device)
    tb_writer.add_graph(model, init_img)

    if args.mode == 'train':
        train_model(train_loader, test_loader, val_loader, model)
    elif args.mode == 'test':
        load_model(test_loader, model, path = "./Model/Best_AlexNet.ckpt")
