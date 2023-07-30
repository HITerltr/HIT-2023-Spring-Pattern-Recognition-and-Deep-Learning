# -*- coding: gbk -*-
import argparse
import numpy as np
import torch
import torch.cuda
from OnlineShoppingNet import RNN
from JenaNet import Jena_Net
from torch.utils.tensorboard import SummaryWriter
from load_online_shopping import Load_Online_Shopping
from load_jena import Load_Jena
from sklearn.metrics import precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt

# �����������
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default = 100)
parser.add_argument('--batch_size', default = 16)
parser.add_argument('--hidden_dim', default = 256)
parser.add_argument('--mode', choices = ['train', 'test'], default = 'train')
parser.add_argument('--best_save_path', default = 'Model/Jena_Best(GRU).ckpt')

args = parser.parse_args()

# ������
batch_size = args.batch_size
epoch = args.epoch

# �趨�豸
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ʵ����tensorboard writer
tb_writer = SummaryWriter(log_dir = "./Runs_jena")
tags = ["Train_Loss", "Ver_Loss", "Learning Rate"]

jena_input_dim = 720
jena_embd_dim = 256
jena_out_dim = 288


def Train_Jena(train_loader, test_loader, ver_loader, model, path = args.best_save_path):
    print("Begin training...")
    best_loss = 1000000000
    # ������ʧ�������Ż���
    loss_func = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.0001)
    # ��ʼѵ��
    for i in range(epoch):
        model.train()
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
        train_loss /= len(train_loader.dataset)
        # ÿ���һ��epoch��ѵ����������֤���ϲ���׼ȷ��
        ver_loss, _, __ = Test_Jena(ver_loader, model)
        # ��ӡ���
        print('Epoch:{}/{}   '.format(i + 1, epoch), end = '')
        print('Train Loss: {:.4f}   Ver Loss: {:.4f}   '.format(train_loss, ver_loss))
        # ����ز���д��tensorboard
        tb_writer.add_scalar(tags[0], train_loss, i)
        tb_writer.add_scalar(tags[1], ver_loss, i)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], i)
        # �������ģ��
        if ver_loss < best_loss:
            best_loss = ver_loss
            torch.save(model.state_dict(), path)
    print("Finished {} Epoch".format(epoch))
    # �ڲ��Լ�����֤���
    test_loss, y_true, y_pred = Test_Jena(test_loader, model)
    print("Test Loss:{:.4f}".format(test_loss))
    plot_results(y_true, y_pred)


def Test_Jena(test_loader, model):
    # ��ʼ��������������ʧ����
    correct = 0.0
    test_loss = 0.0
    loss_f = torch.nn.MSELoss().cuda()
    # ģ����֤
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            # ��������
            data, target = data.to(device), target.to(device)
            # ��������
            output = model(data)
            # ���������ʧ
            test_loss += loss_f(output, target).item() * data.size(0)
            # ��ȡĳ��ʱ���׼ȷ�¶���Ԥ���¶�
            y_true = target.cpu()[0, :]
            y_pred = output.cpu()[0, :]
        test_loss /= len(test_loader.dataset)
    return test_loss, y_true, y_pred


def Load_Model_Jena(test_loader, model, path = args.best_save_path):
    """����ģ�ͽ��в���"""
    model.load_state_dict(torch.load(path))
    model = model.cuda()
    loss, y_true, y_pred = Test_Jena(test_loader, model)
    print("Load model | Loss:{:.4f}".format(loss))
    plot_results(y_true, y_pred)


def Train_Model_Jena(train_loader, test_loader, ver_loader, model):
    # ��ģ�ͷŵ�GPU��
    model = model.cuda()
    Train_Jena(train_loader, test_loader, ver_loader, model)


def plot_results(y_true, y_pred):
    """��ӡ����ͼ"""
    y_true = y_true.tolist()
    y_pred = y_pred.tolist()
    deviation = [y_true[i] - y_pred[i] for i in range(len(y_true))]
    print("Avg error: {:.4f}".format(np.mean(deviation)))
    print("Median error: {:.4f}".format(np.median(deviation)))
    print("Figure shows the prediction on certain period of test set.")
    x = [i for i in range(len(y_true))]
    fig, ax = plt.subplots()  # ����ͼʵ��
    ax.plot(x, y_true, label = 'True', color = '#0000FF')
    ax.plot(x, y_pred, label = 'Prediction', color = '#00FF00')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature')
    ax.set_ylim((-3, 3))
    ax.legend()
    plt.show()


if __name__ == '__main__':
    train_loader, test_loader = Load_Jena(path="Data/Jena_Climate/jena_climate_2009_2016.csv", batch_size = batch_size, shuffle = True)
    ver_loader = test_loader

    # ��������
    model = Jena_Net(jena_embd_dim, args.hidden_dim, jena_out_dim)
    model.to(device)

    if args.mode == 'train':
        Train_Model_Jena(train_loader, test_loader, ver_loader, model)
    elif args.mode == 'test':
        Load_Model_Jena(test_loader, model, path="./Model/Jena_Best(GRU).ckpt")
