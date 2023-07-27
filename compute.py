import numpy as np
from numpy import ndarray


def compute_model_output(x, w, b):
    f"""
    计算模型输出, y-bar = x * w + b
    :param x:
    :type x: ndarray
    :param w:
    :type w: float
    :param b:
    :type b: float
    :return:
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb


def compute_cost(x, y, w, b):
    """
    代价函数
    :param x: x数组
    :type x: ndarray
    :param y: y数组
    :type y: ndarray
    :param w:
    :type w: float
    :param b:
    :type b: float
    :return:
    """

    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost
