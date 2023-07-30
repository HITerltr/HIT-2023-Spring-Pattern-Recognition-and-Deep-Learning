# -*- coding: gbk -*-
import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio
import os
import re

# ʹ�� cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ����ɢ��ͼ
def draw_scatter_chart(data, color, label):
    plt.xlim(-0.25, 1.5)
    plt.ylim(0, 0.8)
    plt.title('Result')
    plt.xlabel("X", fontsize = 14)
    plt.ylabel("Y", fontsize = 14)
    plt.scatter(data[:, 0], data[:, 1], c=color, s = 6, label = label)
    plt.legend(loc = "upper left")


def sort_key(s):
    """�Զ���sort key"""
    c = re.findall('\d+', s)[0]  # ƥ�俪ͷ�������
    return int(c)


# �ϳɶ�ͼ
def show_img(model):
    imgs = []
    temp = os.listdir("./result/{}".format(model))
    img_name = [x for x in temp if 'png' in x]
    img_name.sort(key = sort_key)
    for fileName in img_name:
        imgs.append(imageio.imread(os.path.join('./result/gan', fileName)))
    imageio.mimsave("./GIF/{}.gif".format(model), imgs, fps = 4)
    print("Finished generating gifs for {}".format(model))


if __name__ == '__main__':
    show_img("gan")
    show_img("wgan")
    show_img("wgan_gp")
