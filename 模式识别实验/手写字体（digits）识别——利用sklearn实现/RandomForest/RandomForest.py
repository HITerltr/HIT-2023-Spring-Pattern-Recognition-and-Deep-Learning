# -*- coding: gbk -*-
from learning_curve import plot_learning_curve
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

digits = load_digits()
X = digits.data
y = digits.target


def show(image):
    test = image.reshape((8, 8))  # ��һάת���ɶ�ά���������ܱ���ʾ
    print(test.shape)  # �鿴�Ƿ�Ϊ��ά����
    # print(test)
    plt.imshow(test, cmap = plt.cm.gray)  # ��ʾ��ɫͼ��
    plt.show()


def standard_demo(data):
    transfer = StandardScaler()
    data_new = transfer.fit_transform(data)
    print(data_new)
    return data_new


def pca_demo(data):
    transfer = PCA(n_components = 0.92)
    data_new = transfer.fit_transform(data)
    print(data_new)
    return data_new


def randomforest_demo(data, label):
    # �������ݼ�
    X_train, X_test, y_train, y_test = train_test_split(data, label, random_state = 6)
    # ѵ��ģ��
    estimate = RandomForestClassifier(n_estimators = 100, criterion = 'gini', max_depth = 20000)
    estimate.fit(X_train, y_train)  # ģ�͹������
    # ģ�����������ַ�����1��ֱ�ӱȶ�Ԥ��ֵ����ʵֵ��
    y_predict = estimate.predict(X_test)
    print("ֱ�ӱȶ�Ԥ��ֵ����ʵֵ��\n", y_test == y_predict)
    # 2������׼ȷ��
    score = estimate.score(X_test, y_test)
    print("׼ȷ��Ϊ��\n", score)
    # ����ѧϰ����
    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)  # 10�۽�����֤
    fig, ax = plt.subplots(1, 1, figsize = (10, 6), dpi = 144)
    plot_learning_curve(ax, estimate, "RandomForest Learn Curve",
                        X_train, y_train, ylim = (0.0, 1.01), cv = cv)
    plt.show()
    # ��������
    cm = confusion_matrix(y_test, y_predict)
    print(cm)
    # ���ӻ���ʾ��������
    # annot = True ��ʾ���� ��fmt������ʹ�ÿ�ѧ������������ʾ
    ax = sn.heatmap(cm, annot = True, fmt = '.20g')
    ax.set_title('RandomForest confusion matrix')  # ����
    ax.set_xlabel('predict')  # x��
    ax.set_ylabel('true')  # y��
    plt.show()
    # ���㾫ȷ�����ٻ���
    report = classification_report(y_test, y_predict, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    print(report)


# ������е���ɫ��ť�����нű���
if __name__ == '__main__':
    # �鿴���ݼ�����ͼƬ��ʾ
    print(X.shape, y.shape)
    print(X, y)
    # show(X[1791])
    # ���ݼ�Ԥ������׼����������ά��
    X_new = standard_demo(X)
    X_new = pca_demo(X_new)
    print(X_new.shape)  # ��64ά������40ά
    # ����ѧϰ��ģ
    # ����
    # ģ������
    # ѧϰ����
    randomforest_demo(X_new, y)  # �����Ѿ�׼�����
