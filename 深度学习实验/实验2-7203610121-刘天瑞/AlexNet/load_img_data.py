# -*- coding: gbk -*-
import os
import json
from sklearn.model_selection import train_test_split

filepath = './Data/caltech-101/101_ObjectCategories'


def load_img_data(save = True):
    caltech101_class = [i for i in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, i))]# ��ȡ����ǩ
    caltech101_class.remove("BACKGROUND_Google")
    caltech101_class.sort()# ����
    class_index = dict((v, k) for k, v in enumerate(caltech101_class))

    if save:
        json_file = json.dumps(dict((v, k) for k, v in class_index.items()), indent = 4)
        with open('class.json', 'w') as f:
            f.write(json_file)

    train_path = []
    train_label = []
    test_path = []
    test_label = []
    val_path = []
    val_label = []
    cls_num = []# ����ÿһ���������

    suffix = ['.jpg', '.JPG', '.png', '.PNG']

    # �������ļ���
    for cls in caltech101_class:
        cls_path = os.path.join(filepath, cls)
        imgs = [os.path.join(cls_path, i) for i in os.listdir(cls_path) if
                os.path.splitext(i)[-1] in suffix]
        img_class = class_index[cls]
        cls_num.append(len(imgs))

        # ����ѵ���������Լ��Լ�������
        img_train, others = train_test_split(imgs, test_size = 0.2, shuffle = True)
        img_test, img_val = train_test_split(others, test_size = 0.5, shuffle = True)

        for img in imgs:
            if img in img_train:
                train_path.append(img)
                train_label.append(img_class)
            elif img in img_test:
                test_path.append(img)
                test_label.append(img_class)
            elif img in img_val:
                val_path.append(img)
                val_label.append(img_class)

    return train_path, train_label, test_path, test_label, val_path, val_label

