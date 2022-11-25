# -*- coding: utf-8 -*-
# @Time    : 2022/11/19 21:29
# @Author  : SourDumplings
# @Email   : changzheng300@foxmail.com
# @File    : main.py


'''
参考：https://www.zhihu.com/collection/697302383
'''

import numpy as np


def sigmoid(x, deriv=False):
    if deriv:
        # Sigmoid 函数的导数
        return x * (1 - x)
    else:
        # Sigmoid 函数，将所有的实数 x 映射到 (0, 1) 内
        # 参考：https://baike.baidu.com/item/Sigmoid%E5%87%BD%E6%95%B0/7981407
        return 1 / (1 + np.exp(-x))


def train(X, Y):
    # 设置随机数种子，使得每次运行程序所产生的的随机数序列固定，便于程序调试
    np.random.seed(1)

    # synapses 突触矩阵，即神经元的连接处
    # input、单层神经元、output 三层，共需要两层的突触连接
    syn0 = 2 * np.random.random((3, 4)) - 1  # 3 * 4 的矩阵
    syn1 = 2 * np.random.random((4, 1)) - 1  # 4 * 1 的矩阵

    # training step
    for j in range(200000):
        l0 = X  # 第一层 (input)
        l1 = sigmoid(np.dot(l0, syn0))  # 第二层
        l2 = sigmoid(np.dot(l1, syn1))  # 第三层 (output 的预测)
        error = (1 / 2) * (Y - l2) * (Y - l2)
        if j % 10000 == 0:
            print("Error in %d iteration for training: %s" % (j, str(np.mean(error))))

        # 反向传播算法实现梯度下降，参考：https://blog.csdn.net/ft_sunshine/article/details/90221691
        # 这里 delta 计算本质是链导法则
        l2_delta = -(Y - l2) * sigmoid(l2, True)
        l1_delta = l2_delta.dot(syn1.T) * sigmoid(l1, True)
        # update weights，更新突触权重，实现梯度下降
        syn1 -= l1.T.dot(l2_delta)
        syn0 -= l0.T.dot(l1_delta)

    print("Training finish!")
    print("syn0:")
    print(syn0)
    print("syn1:")
    print(syn1)
    return syn0, syn1


def test(X, syn0, syn1):
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))  # 第二层
    l2 = sigmoid(np.dot(l1, syn1))  # 第三层 (output 的预测)
    return l2


def main():
    # input data
    # 每一列代表一个神经元
    # 故总共有 3 个神经元，各自接受 4 个输入数据
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

    # output data
    Y = np.array([[0],
                  [1],
                  [0],
                  [1]])

    syn0, syn1 = train(X, Y)
    res = test(X, syn0, syn1)
    print("test res:")
    print(res)


if __name__ == '__main__':
    main()
