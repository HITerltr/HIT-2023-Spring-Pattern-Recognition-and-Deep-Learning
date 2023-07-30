# -*- coding: gbk -*-
from re import T
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import draw
import argparse
import os

# �����������
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default = 256)
parser.add_argument('--epoch', default = 320)
parser.add_argument('--lr', default = 0.00005)
parser.add_argument('--train_size', default = 7000, help = 'ѵ�����Ĵ�С')
parser.add_argument('--input_size', default = 2, help = '���������������ά��')
parser.add_argument('--clamp', default = 0.1, help = 'WGAN��Ȩֵ����')
args = parser.parse_args()

# ������
model_name = 'wgan_gp'
batch_size = args.batch_size
epoch = args.epoch
train_size = args.train_size
input_size = args.input_size
lr = args.lr
clamp = args.clamp

# �趨�豸
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# �������ݼ�
data = loadmat("./points.mat")['xx']
np.random.shuffle(data)
# ������ݼ�
train_set = data[:train_size]
test_set = data[train_size:]


# ��������
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(64, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(512, 2)
        )

    def Forward(self, x):
        return self.net(x)


# �б���
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(64, 256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(256, 1),
        )

    def Forward(self, x):
        return self.net(x)


# �������������б���������
D = Discriminator().to(device)
G = Generator().to(device)
# �����Ż���
optimizer_D = torch.optim.RMSprop(D.parameters(), lr = lr)
optimizer_G = torch.optim.RMSprop(G.parameters(), lr = lr)


def Train():
    for ep in range(epoch):
        loss_D = 0
        loss_G = 0
        for i in range(int(train_size / batch_size)):
            real = torch.from_numpy(train_set[i * batch_size: (i + 1) * batch_size]).float().to(device)
            G_input = torch.randn(batch_size, input_size).to(device)
            fake = G(G_input)
            # ����Discriminator�б�fake�ĸ���
            outFake = D(fake)
            # ����Generator��loss����ʹ��log��
            loss_G = -torch.mean(outFake)
            # ����������
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            # ����real�ĸ���
            outReal = D(real)
            # ���¼���
            outFake = D(fake.detach())
            # ����ͷ���
            gradient_penalty = compute_gradient_penalty(real.detach(), fake.detach())
            # ����Discriminator��loss����ʹ��log��
            loss_D = (torch.mean(outFake - outReal) + 0.001 * gradient_penalty)
            # �����б���
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            # # ����D�в�����ֵ
            # for p in D.parameters():
            #     p.data.clamp_(-clamp, clamp)
        print("epoch: {:d}     d_loss: {:.3f}     g_loss: {:.3f} ".format(ep + 1, loss_D, loss_G))
        if (ep + 1) % 10 == 0:
            Test_Generator()
            plt.savefig(os.path.join('./result', model_name, str(ep + 1)))
            plt.cla()


def Test_Generator():
    """����Generator"""
    G_input = torch.randn(1200, input_size).to(device)
    G_out = G(G_input)
    G_data = np.array(G_out.cpu().data)
    # �������Լ��ĵ�ֲ�������������ĵ�ֲ�
    draw.draw_scatter(test_set, '#228B22', 'Original')
    draw.draw_scatter(G_data, '#1E90FF', 'Generated')
    return


def Compute_Gradient_Penalty(real_samples, fake_samples):
    """����gradient_penalty"""
    Tensor = torch.FloatTensor
    alpha = torch.rand(real_samples.shape).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(device)
    d_interpolates = D(interpolates)
    fake = autograd.Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad = False)
    gradients = autograd.grad(
        outputs = d_interpolates,
        inputs=interpolates,
        grad_outputs = fake.to(device),
        create_graph = True,
        retain_graph = True,
        only_inputs = True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


if __name__ == '__main__':
    Train()
    Test_Generator()
