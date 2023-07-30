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
    test = image.reshape((8, 8))  # 从一维转换成二维，这样才能被显示
    print(test.shape)  # 查看是否为二维数组
    # print(test)
    plt.imshow(test, cmap = plt.cm.gray)  # 显示灰色图像
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
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data, label, random_state = 6)
    # 训练模型
    estimate = RandomForestClassifier(n_estimators = 100, criterion = 'gini', max_depth = 20000)
    estimate.fit(X_train, y_train)  # 模型构建完毕
    # 模型评估的两种方法：1：直接比对预测值与真实值；
    y_predict = estimate.predict(X_test)
    print("直接比对预测值与真实值：\n", y_test == y_predict)
    # 2：计算准确率
    score = estimate.score(X_test, y_test)
    print("准确率为：\n", score)
    # 绘制学习曲线
    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)  # 10折交叉验证
    fig, ax = plt.subplots(1, 1, figsize = (10, 6), dpi = 144)
    plot_learning_curve(ax, estimate, "RandomForest Learn Curve",
                        X_train, y_train, ylim = (0.0, 1.01), cv = cv)
    plt.show()
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_predict)
    print(cm)
    # 可视化显示混淆矩阵
    # annot = True 显示数字 ，fmt参数不使用科学计数法进行显示
    ax = sn.heatmap(cm, annot = True, fmt = '.20g')
    ax.set_title('RandomForest confusion matrix')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.show()
    # 计算精确率与召回率
    report = classification_report(y_test, y_predict, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    print(report)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # 查看数据集，以图片显示
    print(X.shape, y.shape)
    print(X, y)
    # show(X[1791])
    # 数据集预处理（标准化、特征降维）
    X_new = standard_demo(X)
    X_new = pca_demo(X_new)
    print(X_new.shape)  # 从64维降到了40维
    # 机器学习建模
    # 调参
    # 模型评估
    # 学习曲线
    randomforest_demo(X_new, y)  # 数据已经准备完毕
