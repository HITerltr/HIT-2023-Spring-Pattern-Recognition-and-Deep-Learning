# -*- coding: gbk -*-
import argparse
import numpy as np
import torch
import torch.cuda
from OnlineShoppingNet import RNN, LSTM, Bi_LSTM, GRU
from JenaNet import Jena_Net
from torch.utils.tensorboard import SummaryWriter
from load_online_shopping import Load_Online_Shopping
from load_jena import Load_Jena
from sklearn.metrics import classification_report

# �����������
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default = 100)
parser.add_argument('--max_len', default = 50)
parser.add_argument('--batch_size', default = 64)
parser.add_argument('--hidden_dim', default = 256)
parser.add_argument('--mode', choices = ['train', 'test'], default = 'test')
parser.add_argument('--model', choices = ['RNN', 'LSTM', 'GRU', 'Bi_LSTM'], default = 'Bi_LSTM')
parser.add_argument('--npz_path', default = 'Data/Online_Shopping/w2v.npz')
parser.add_argument('--best_save_path', default = 'Model/OS_Bi_LSTM.ckpt')

args = parser.parse_args()

# ������
max_len = args.max_len
batch_size = args.batch_size
epoch = args.epoch

# �趨�豸
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ʵ����tensorboard writer
tb_writer = SummaryWriter(log_dir = "./Runs_Online_Shopping")
tags = ["Train_loss", "Ver_loss", "Accuracy", "Learning Rate"]

# ���ݱ�ǩ
data_class = {"�鼮": 0, "ƽ��": 1, "�ֻ�": 2, "ˮ��": 3, "ϴ��ˮ": 4, "��ˮ��": 5, "��ţ": 6, "�·�": 7, "�����": 8, "�Ƶ�": 9}


def Train(train_loader, test_loader, ver_loader, model, path = args.best_save_path):
    print("Begin training...")
    # �������accuracy
    best_accuracy = 0
    # ����loss��accuracyֵ����ͼ�ã�
    # loss_list = []
    # acc_list = []
    # ������ʧ�������Ż���
    loss_function = torch.nn.CrossEntropyLoss().cuda()
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
            loss = loss_function(output, target)
            # ���򴫲�
            loss.backward()
            # ������������������
            optimizer.step()
            # ������ʧ
            train_loss += loss.item() * data.size(0)
        train_loss /= len(train_loader.dataset)
        # ÿ���һ��epoch��ѵ����������֤���ϲ���׼ȷ��
        accuracy, ver_loss, _, __ = Test(ver_loader, model)
        # ��ӡ���
        print('Epoch:{}/{}   '.format(i + 1, epoch), end = '')
        print('Train Loss: {:.4f}   Ver Loss: {:.4f}   '.format(train_loss, ver_loss), end = '')
        print('Accuracy:{:.4f}'.format(accuracy))
        # ����ز���д��tensorboard
        tb_writer.add_scalar(tags[0], train_loss, i)
        tb_writer.add_scalar(tags[1], ver_loss, i)
        tb_writer.add_scalar(tags[2], accuracy, i)
        tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], i)
        # �������ģ��
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), path)
    print("Finished {} Epoch".format(epoch))
    # �ڲ��Լ�����֤���
    test_accuracy, test_loss, y_true, y_pred = Test(test_loader, model)
    print("Test Accuracy:{:.4f}   Test Loss:{:.4f}".format( test_accuracy, test_loss))
    print("Online shopping evaluation results:")
    # print(precision_recall_fscore_support(y_true, y_pred, average=None, labels=[i for i in range(0, 10)]))
    print(classification_report(y_true, y_pred, labels = [i for i in range(0, 10)],
                                target_names = ["�鼮", "ƽ��", "�ֻ�", "ˮ��", "ϴ��ˮ", "��ˮ��", "��ţ", "�·�", "�����", "�Ƶ�"]))


def Test(test_loader, model):
    # ��ʼ��������������ʧ����
    correct = 0.0
    test_loss = 0.0
    y_true = np.zeros((1,), dtype = 'int')
    y_pred = np.zeros((1,), dtype = 'int')
    loss_function = torch.nn.CrossEntropyLoss().cuda()
    # ģ����֤
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            y_true = np.concatenate((y_true, target.numpy()))
            # ��������
            data, target = data.to(device), target.to(device)
            # ��������
            output = model(data)
            # ���������ʧ
            test_loss += loss_function(output, target).item() * data.size(0)
            # �ҵ�����ֵ�����±�
            pred = output.argmax(dim = 1)
            y_pred = np.concatenate((y_pred, pred.cpu().numpy()))
            # �ۼ���ȷ��
            correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
        y_true = y_true[1:]
        y_pred = y_pred[1:]
    return accuracy, test_loss, y_true, y_pred


def Load_Model(test_loader, model, path = args.best_save_path):
    """����ģ�ͽ��в���"""
    model.load_state_dict(torch.load(path))
    model = model.cuda()
    accuracy, test_loss, y_true, y_pred = Test(test_loader, model)
    print("Load model | Test_accuracy:{:.4f}  Test_loss:{:.4f}".format(accuracy,test_loss))


def Train_Model(train_loader, test_loader, ver_loader, model):
    # ��ģ�ͷŵ�GPU��
    model = model.cuda()
    Train(train_loader, test_loader, ver_loader, model)


if __name__ == '__main__':
    loaders = Load_Online_Shopping(path = "Data/Online_Shopping", max_len = max_len, batch_size = batch_size, shuffle = True)
    train_loader = loaders[0]
    ver_loader = loaders[1]
    test_loader = loaders[2]

    # ��ȡ����������
    print("Load .npz file...")
    loaded = np.load(args.npz_path)
    embeddings = torch.FloatTensor(loaded['embeddings'])
    embedding_dim = embeddings.shape[1]
    a = embeddings.dim()
    print("-- have finished!")

    # ��������
    if args.model == 'RNN':
        model = RNN(embeddings, embedding_dim, args.hidden_dim, len(data_class))
    elif args.model == 'LSTM':
        model = LSTM(embeddings, embedding_dim, args.hidden_dim, len(data_class))
    elif args.model == 'GRU':
        model = GRU(embeddings, embedding_dim, args.hidden_dim, len(data_class))
    elif args.model == 'Bi_LSTM':
        model = Bi_LSTM(embeddings, embedding_dim, args.hidden_dim, len(data_class))
    model.to(device)
    # ��ģ�ͽṹд��tensorboard
    # init_img = torch.zeros((1, 100, 300), device = device)
    # tb_writer.add_graph(model, init_img)

    if args.mode == 'train':
        Train_Model(train_loader, test_loader, ver_loader, model)
    elif args.mode == 'test':
        Load_Model(test_loader, model, path = "./Model/OS_{}.ckpt".format(args.model))
