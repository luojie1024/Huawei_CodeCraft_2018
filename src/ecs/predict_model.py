# -*- coding: utf-8 -*-
import copy
import math
import random
import martix_utils as vt
import LSTM as lstmcl
import time as t
import datetime
import math

from const_map import *

# 加入随机数
is_noise = 0


def predict_model1(his_data, date_range_size, vm_type):  # 简单滑动平均法
    '''
    预测方案 1 指数滑动平均
    :param his_data: 真实的历史数据出现次数表
    :param date_range_size: 需要预测的长度
    :return: 返回结果
    '''

    # sigma = 0.5

    # 衰减值0.21
    alpha = 0.21
    # 历史天数
    chis_data = copy.deepcopy(his_data['value'])
    # 历史天数
    cal_len = len(chis_data)
    temp_reuslt = 0.0
    result = []

    for rept in range(date_range_size):  # 预测天数范围
        temp_value = 0.0
        # 遍历预测值
        for i in range(len(chis_data)):
            temp_value += chis_data[len(chis_data) - i - 1] * alpha * pow(1 - alpha, i)
        chis_data.append(temp_value)
        temp_reuslt += temp_value

    # noise = random.gauss(0, sigma)
    # noise = math.fabs(noise)
    # # 求一个浮点数的地板，就是求一个最接近它的整数 ceil向上取整
    # result.append(int(math.floor(temp_reuslt+noise)))
    result.append(int(math.floor(temp_reuslt)))
    return result

def predict_model2(his_data, date_range_size, vm_type):
    '''
    预测方案 2 指数滑动平均
    :param his_data: 真实的历史数据出现次数表
    :param date_range_size: 需要预测的长度
    :return: 返回结果
    '''

    # sigma = 0.5

    # 衰减值0.21
    alpha = 0.21
    # 历史天数
    chis_data = copy.deepcopy(his_data['value'])
    # 历史天数
    cal_len = len(chis_data)
    temp_reuslt = 0.0
    result = []

    for rept in range(date_range_size):  # 预测天数范围
        temp_value = 0.0
        # 遍历预测值
        for i in range(len(chis_data)):
            temp_value += chis_data[len(chis_data) - i - 1] * alpha * pow(1 - alpha, i)
        chis_data.append(temp_value)
        temp_reuslt += temp_value

    # noise = random.gauss(0, sigma)
    # noise = math.fabs(noise)
    # # 求一个浮点数的地板，就是求一个最接近它的整数 ceil向上取整
    # result.append(int(math.floor(temp_reuslt+noise)))
    result.append(int(math.floor(temp_reuslt)))
    return result


def predict_model3(his_data, date_range_size, vm_type):  # 简单滑动平均法
    '''
    预测方案 3 指数滑动平均
    :param his_data: 真实的历史数据出现次数表
    :param date_range_size: 需要预测的长度
    :return: 返回结果
    '''

    # sigma = 0.5

    # 衰减值0.21
    alpha = 0.21
    # 历史天数
    chis_data = copy.deepcopy(his_data['value'])
    # 历史天数
    cal_len = len(chis_data)
    temp_reuslt = 0.0
    result = []

    for rept in range(date_range_size):  # 预测天数范围
        temp_value = 0.0
        # 遍历预测值
        for i in range(len(chis_data)):
            temp_value += chis_data[len(chis_data) - i - 1] * alpha * pow(1 - alpha, i)
        chis_data.append(temp_value)
        temp_reuslt += temp_value

    # noise = random.gauss(0, sigma)
    # noise = math.fabs(noise)
    # # 求一个浮点数的地板，就是求一个最接近它的整数 ceil向上取整
    # result.append(int(math.floor(temp_reuslt+noise)))
    result.append(int(math.floor(temp_reuslt)))
    return result


def predict_model4(his_data, date_range_size, vm_type):  # 霍尔特线性趋势法
    '''
    预测方案 4 霍尔特线性趋势法
    :param his_data: 真实的历史数据出现次数表
    :param date_range_size: 需要预测的长度
    :return: 返回结果
    '''
    # 历史天数
    chis_data = copy.deepcopy(his_data['value'])
    # 历史天数
    cal_len = len(chis_data)
    temp_reuslt = 0.0
    result = []
    #
    sigma = 0.2

    # 衰减值
    alpha = 0.2
    # 趋势
    beta = 0.0
    # 权重 75.21
    h = 1.75

    y_hot_t = 0.0
    l_t = 0.2
    b_t = 0.2

    # 初始trend
    pre_b_t = 0.0
    # 初始化level
    pre_l_t = 0.0

    for rept in range(date_range_size):  # 预测天数范围

        # 遍历历史记录
        for i in range(1, len(chis_data)):  # t+1开始
            # 更新level trend
            pre_l_t = l_t
            pre_b_t = b_t
            # step1 computer level
            l_t = alpha * chis_data[i - 1] + (1 - alpha) * (pre_l_t + pre_b_t)
            # step2 computer trend
            b_t = beta * (l_t - pre_l_t) + (1 - beta) * b_t
            # step3
            y_hot_t = l_t + h * b_t
            if y_hot_t < 0:
                y_hot_t = 0
        # 追加到历史表中
        chis_data.append(y_hot_t)
        # 保存结果
        temp_reuslt += y_hot_t
    noise = random.gauss(0, sigma)
    noise = math.fabs(noise)
    # 求一个浮点数的地板，就是求一个最接近它的整数 ceil向上取整
    result.append(int(math.floor(temp_reuslt) + noise))
    return result

def predict_model5(his_data, date_range_size, vm_type):  # 霍尔特线性趋势法
    '''
    预测方案 十 霍尔特线性趋势法
    :param his_data: 真实的历史数据出现次数表
    :param date_range_size: 需要预测的长度
    :return: 返回结果
    '''
    # 历史天数
    chis_data = copy.deepcopy(his_data['value'])
    # 历史天数
    cal_len = len(chis_data)
    temp_reuslt = 0.0
    result = []
    #
    sigma = 0.2

    # 衰减值
    alpha = 0.2
    # 趋势
    beta = 0.0
    # 权重 75.21
    h = 1.75

    y_hot_t = 0.0
    l_t = 0.2
    b_t = 0.2

    # 初始trend
    pre_b_t = 0.0
    # 初始化level
    pre_l_t = 0.0

    for rept in range(date_range_size):  # 预测天数范围

        # 遍历历史记录
        for i in range(1, len(chis_data)):  # t+1开始
            # 更新level trend
            pre_l_t = l_t
            pre_b_t = b_t
            # step1 computer level
            l_t = alpha * chis_data[i - 1] + (1 - alpha) * (pre_l_t + pre_b_t)
            # step2 computer trend
            b_t = beta * (l_t - pre_l_t) + (1 - beta) * b_t
            # step3
            y_hot_t = l_t + h * b_t
            if y_hot_t < 0:
                y_hot_t = 0
        # 追加到历史表中
        chis_data.append(y_hot_t)
        # 保存结果
        temp_reuslt += y_hot_t
    noise = random.gauss(0, sigma)
    noise = math.fabs(noise)
    # 求一个浮点数的地板，就是求一个最接近它的整数 ceil向上取整
    result.append(int(math.floor(temp_reuslt) + noise))
    return result


def predict_model6(his_data, date_range_size, vm_type):  # 霍尔特线性趋势法
    '''
    预测方案 6 霍尔特线性趋势法
    :param his_data: 真实的历史数据出现次数表
    :param date_range_size: 需要预测的长度
    :return: 返回结果
    '''
    # 历史天数
    chis_data = copy.deepcopy(his_data['value'])
    # 历史天数
    cal_len = len(chis_data)
    temp_reuslt = 0.0
    result = []
    #
    sigma = 0.2

    # 衰减值
    alpha = 0.2
    # 趋势
    beta = 0.0
    # 权重 75.21
    h = 1.75

    y_hot_t = 0.0
    l_t = 0.2
    b_t = 0.2

    # 初始trend
    pre_b_t = 0.0
    # 初始化level
    pre_l_t = 0.0

    for rept in range(date_range_size):  # 预测天数范围

        # 遍历历史记录
        for i in range(1, len(chis_data)):  # t+1开始
            # 更新level trend
            pre_l_t = l_t
            pre_b_t = b_t
            # step1 computer level
            l_t = alpha * chis_data[i - 1] + (1 - alpha) * (pre_l_t + pre_b_t)
            # step2 computer trend
            b_t = beta * (l_t - pre_l_t) + (1 - beta) * b_t
            # step3
            y_hot_t = l_t + h * b_t
            if y_hot_t < 0:
                y_hot_t = 0
        # 追加到历史表中
        chis_data.append(y_hot_t)
        # 保存结果
        temp_reuslt += y_hot_t
    noise = random.gauss(0, sigma)
    noise = math.fabs(noise)
    # 求一个浮点数的地板，就是求一个最接近它的整数 ceil向上取整
    result.append(int(math.floor(temp_reuslt) + noise))
    return result


def predict_model7(his_data,  # 某种类型的虚拟机的历史数据
                   date_range_size, vm_type):  # 需要预测的长度

    '''
    预测方案七 对若干星期前同一天数据求平均
    '''

    n = 10  # 边长数10
    sigma = 0.5

    back_week = 1 #1
    chis_data = copy.deepcopy(his_data['value'])
    cal_len = len(chis_data)

    result = []
    for rept in range(date_range_size):  # 预测天数范围
        day_avage = 0.0
        cot_week = 0
        for i in range(1, back_week + 1):
            index = i * 7
            if index <= cal_len:
                day_tmp = chis_data[-index] * n
                cot_day = n
                cot_week += 1
                for j in range(1, n):
                    tmp = (n - j) / 2.0
                    day_tmp += chis_data[-index + j] * tmp
                    cot_day += tmp
                    if index + j <= cal_len:
                        day_tmp += chis_data[-index - j] * tmp
                        cot_day += tmp
                    else:
                        continue
                day_avage += day_tmp / cot_day
            else:
                break
        if cot_week != 0:  # 直接平均  --> 改进成指数平均
            day_avage = day_avage * 1.0 / cot_week  # 注意报错
        noise = random.gauss(0, sigma)
        noise = math.fabs(noise)
        day_avage = int(math.ceil(day_avage + noise))
        chis_data.append(day_avage)
        result.append(day_avage)

    return result


def predict_model8(his_data,  # 某种类型的虚拟机的历史数据
                   date_range_size, vm_type):  # 需要预测的长度

    #无noise

    n = 10  # 边长数10
    sigma = 0.5

    beta = 3.0
    back_week = 1 #1 2
    chis_data = copy.deepcopy(his_data['value'])
    cal_len = len(chis_data)

    result = []
    temp_result=0
    for rept in range(date_range_size):  # 预测天数范围
        day_avage = 0.0
        cot_week = 0
        for i in range(1, back_week + 1):
            index = i * 7
            if index <= cal_len:
                day_tmp = chis_data[-index] * n
                cot_day = n
                cot_week += 1
                for j in range(1, n):
                    tmp = (n - j) / beta
                    day_tmp += chis_data[-index + j] * tmp
                    cot_day += tmp
                    if index + j <= cal_len:
                        day_tmp += chis_data[-index - j] * tmp
                        cot_day += tmp
                    else:
                        continue
                day_avage += day_tmp / cot_day
            else:
                break
        if cot_week != 0:  # 直接平均  --> 改进成指数平均
            day_avage = day_avage * 1.0 / cot_week  # 注意报错
        # noise = random.gauss(0, sigma)
        # noise = math.fabs(noise)
        # day_avage = int(math.ceil(day_avage + noise))
        day_avage = int(math.ceil(day_avage))
        chis_data.append(day_avage)
        temp_result+=day_avage
    result.append(temp_result)

    return result


def predict_model9(his_data, date_range_size, vm_type):  # 简单滑动平均法
    '''
    预测方案 九 指数滑动平均
    :param his_data: 真实的历史数据出现次数表
    :param date_range_size: 需要预测的长度
    :return: 返回结果
    '''

    # sigma = 0.5

    # 衰减值0.21
    alpha = 0.21
    # 历史天数
    chis_data = copy.deepcopy(his_data['value'])
    # 历史天数
    cal_len = len(chis_data)
    temp_reuslt = 0.0
    result = []

    for rept in range(date_range_size):  # 预测天数范围
        temp_value = 0.0
        # 遍历预测值
        for i in range(len(chis_data)):
            temp_value += chis_data[len(chis_data) - i - 1] * alpha * pow(1 - alpha, i)
        chis_data.append(temp_value)
        temp_reuslt += temp_value

    # noise = random.gauss(0, sigma)
    # noise = math.fabs(noise)
    # # 求一个浮点数的地板，就是求一个最接近它的整数 ceil向上取整
    # result.append(int(math.floor(temp_reuslt+noise)))
    result.append(int(math.floor(temp_reuslt)))
    return result


def predict_model10(his_data, date_range_size, vm_type):  # 霍尔特线性趋势法
    '''
    预测方案 十 霍尔特线性趋势法
    :param his_data: 真实的历史数据出现次数表
    :param date_range_size: 需要预测的长度
    :return: 返回结果
    '''
    # 历史天数
    chis_data = copy.deepcopy(his_data['value'])
    # 历史天数
    cal_len = len(chis_data)
    temp_reuslt = 0.0
    result = []
    #
    sigma = 0.2

    # 衰减值
    alpha = 0.2
    # 趋势
    beta = 0.0
    # 权重 75.21
    h = 1.75

    y_hot_t = 0.0
    l_t = 0.2
    b_t = 0.2

    # 初始trend
    pre_b_t = 0.0
    # 初始化level
    pre_l_t = 0.0

    for rept in range(date_range_size):  # 预测天数范围

        # 遍历历史记录
        for i in range(1, len(chis_data)):  # t+1开始
            # 更新level trend
            pre_l_t = l_t
            pre_b_t = b_t
            # step1 computer level
            l_t = alpha * chis_data[i - 1] + (1 - alpha) * (pre_l_t + pre_b_t)
            # step2 computer trend
            b_t = beta * (l_t - pre_l_t) + (1 - beta) * b_t
            # step3
            y_hot_t = l_t + h * b_t
            if y_hot_t < 0:
                y_hot_t = 0
        # 追加到历史表中
        chis_data.append(y_hot_t)
        # 保存结果
        temp_reuslt += y_hot_t
    noise = random.gauss(0, sigma)
    noise = math.fabs(noise)
    # 求一个浮点数的地板，就是求一个最接近它的整数 ceil向上取整
    result.append(int(math.floor(temp_reuslt) + noise))
    return result


################Holt-Winters#########################
# 用例01  76.68  小于三种类型
def predict_model11(his_data, date_range_size, vm_type):  # Holt-Winters法
    '''
    预测方案 十一 Holt-Winters
    :param his_data: 真实的历史数据出现次数表
    :param date_range_size: 需要预测的长度
    :param k:跨度天数
    :return: 返回结果
    '''
    Y = copy.deepcopy(his_data['value'])

    k = 1

    temp_reuslt = 0.0
    result = []

    # 衰减值 0218
    alpha = 0.220
    # 趋势
    beta = 0.000
    # 季节 0.21
    gamma = 0.215
    # 季度周期长度
    s = 7

    l_t = []
    b_t = []
    s_t = []

    # 初始trend
    pre_b_t = 0.0
    # 初始化level
    pre_l_t = 0.0
    # 初始化seasonal
    pre_s_t = 0.0

    # 初始化第一天的季动
    l_t.append(pre_l_t)
    b_t.append(pre_b_t)
    s_t.append(pre_s_t)

    # 在首部填充一位数据初始
    Y.insert(0, 0.0)

    # 用历史记录训练初始化参数
    for t in range(1, len(Y)):  # 当前是t时刻
        # 参数
        if (t - s) < len(s_t):  # 初始季动可能越界,越界则用上一个填充
            l_t.append(alpha * (Y[t] - s_t[t - 1]) + (1 - alpha) * (l_t[t - 1] + b_t[t - 1]))
        else:
            l_t.append(alpha * (Y[t] - s_t[t - s]) + (1 - alpha) * (l_t[t - 1] + b_t[t - 1]))

        b_t.append(beta * (l_t[t] - l_t[t - 1]) + (1 - beta) * b_t[t - 1])

        if (t - s) < len(s_t):  # 初始季动可能越界,越界则用上一个填充
            s_t.append(gamma * (Y[t] - l_t[t]) + (1 - gamma) * s_t[t - 1])
        else:
            s_t.append(gamma * (Y[t] - l_t[t]) + (1 - gamma) * s_t[t - s])

    t = len(l_t) - 1

    # 预测要预测的时间k为相隔多少天,相连预测数据相隔k=1
    for h in range(k, date_range_size + k):
        # 追加到历史表中
        # temp_Y = l_t[t] + h*b_t[t] + s_t[t - s+1+((h-1)%s)]

        temp_Y = l_t[t] + h * b_t[t] + s_t[t - s + h]
        # # 如果小于0 置为零
        # if temp_Y < 0:
        #     temp_Y = 0
        # 保存结果
        temp_reuslt += temp_Y
        # 求一个浮点数的地板，就是求一个最接近它的整数 ceil向上取整
        if temp_reuslt < 0:
            temp_reuslt = 0

    # 结果修正
    modify = VM_TYPE_MODIFY1[vm_type]
    temp_reuslt = int(math.floor(temp_reuslt) + modify)
    if temp_reuslt < 0:
        temp_reuslt = 0
    result.append(temp_reuslt)
    return result


# 用例02  76.147
def predict_model12(his_data, date_range_size, vm_type):  # Holt-Winters法
    '''
    预测方案 十二 Holt-Winters
    :param his_data: 真实的历史数据出现次数表
    :param date_range_size: 需要预测的长度
    :param k:跨度天数
    :return: 返回结果
    '''
    # 历史天数
    Y = copy.deepcopy(his_data['value'])
    k = 1

    temp_reuslt = 0.0
    result = []

    # 衰减值0.195
    alpha = 0.195
    # 趋势
    beta = 0.000
    # 季节
    gamma = 0.21
    # 季度周期长度
    s = 7

    l_t = []
    b_t = []
    s_t = []

    # 初始trend
    pre_b_t = 0.0
    # 初始化level
    pre_l_t = 0.0
    # 初始化seasonal
    pre_s_t = 0.0

    # 初始化第一天的季动
    l_t.append(pre_l_t)
    b_t.append(pre_b_t)
    s_t.append(pre_s_t)

    # 在首部填充一位数据初始
    Y.insert(0, 0.0)

    # 用历史记录训练初始化参数
    for t in range(1, len(Y)):  # 当前是t时刻
        # 参数
        if (t - s) < len(s_t):  # 初始季动可能越界,越界则用上一个填充
            l_t.append(alpha * (Y[t] - s_t[t - 1]) + (1 - alpha) * (l_t[t - 1] + b_t[t - 1]))
        else:
            l_t.append(alpha * (Y[t] - s_t[t - s]) + (1 - alpha) * (l_t[t - 1] + b_t[t - 1]))

        b_t.append(beta * (l_t[t] - l_t[t - 1]) + (1 - beta) * b_t[t - 1])

        if (t - s) < len(s_t):  # 初始季动可能越界,越界则用上一个填充
            s_t.append(gamma * (Y[t] - l_t[t]) + (1 - gamma) * s_t[t - 1])
        else:
            s_t.append(gamma * (Y[t] - l_t[t]) + (1 - gamma) * s_t[t - s])

    t = len(l_t) - 1

    # 预测要预测的时间k为相隔多少天,相连预测数据相隔k=1
    for h in range(k, date_range_size + k):
        # 追加到历史表中
        # temp_Y = l_t[t] + h*b_t[t] + s_t[t - s+1+((h-1)%s)]

        temp_Y = l_t[t] + h * b_t[t] + s_t[t - s + h]
        # # 如果小于0 置为零
        # if temp_Y < 0:
        #     temp_Y = 0
        # 保存结果
        temp_reuslt += temp_Y
        # 求一个浮点数的地板，就是求一个最接近它的整数 ceil向上取整
        if temp_reuslt < 0:
            temp_reuslt = 0

    # 结果修正
    modify = VM_TYPE_MODIFY2[vm_type]
    temp_reuslt = int(math.floor(temp_reuslt) + modify)
    if temp_reuslt < 0:
        temp_reuslt = 0
    result.append(temp_reuslt)
    return result


# 用例03 小于三种类型
def predict_model13(his_data, date_range_size, vm_type):  # Holt-Winters法
    '''
    预测方案 十三 Holt-Winters
    :param his_data: 真实的历史数据出现次数表
    :param date_range_size: 需要预测的长度
    :param k:跨度天数
    :return: 返回结果
    '''
    # 历史天数
    Y = copy.deepcopy(his_data['value'])
    k = 1

    temp_reuslt = 0.0
    result = []

    # 衰减值0.186
    alpha = 0.19
    # 趋势0.0
    beta = 0.0
    # 季节 0.185
    gamma = 0.185
    # 季度周期长度 7
    s = 7

    l_t = []
    b_t = []
    s_t = []

    # 初始trend
    pre_b_t = 0.0
    # 初始化level
    pre_l_t = 0.0
    # 初始化seasonal
    pre_s_t = 0.0

    # 初始化第一天的季动
    l_t.append(pre_l_t)
    b_t.append(pre_b_t)
    s_t.append(pre_s_t)

    # 在首部填充一位数据初始
    Y.insert(0, 0.0)

    # 用历史记录训练初始化参数
    for t in range(1, len(Y)):  # 当前是t时刻
        # 参数
        if (t - s) < len(s_t):  # 初始季动可能越界,越界则用上一个填充
            l_t.append(alpha * (Y[t] - s_t[t - 1]) + (1 - alpha) * (l_t[t - 1] + b_t[t - 1]))
        else:
            l_t.append(alpha * (Y[t] - s_t[t - s]) + (1 - alpha) * (l_t[t - 1] + b_t[t - 1]))

        b_t.append(beta * (l_t[t] - l_t[t - 1]) + (1 - beta) * b_t[t - 1])

        if (t - s) < len(s_t):  # 初始季动可能越界,越界则用上一个填充
            s_t.append(gamma * (Y[t] - l_t[t]) + (1 - gamma) * s_t[t - 1])
        else:
            s_t.append(gamma * (Y[t] - l_t[t]) + (1 - gamma) * s_t[t - s])

    t = len(l_t) - 1

    # 预测要预测的时间k为相隔多少天,相连预测数据相隔k=1
    for h in range(k, date_range_size + k):
        # 追加到历史表中
        # temp_Y = l_t[t] + h*b_t[t] + s_t[t - s+1+((h-1)%s)]

        temp_Y = l_t[t] + h * b_t[t] + s_t[t - s + h]
        # # 如果小于0 置为零
        # if temp_Y < 0:
        #     temp_Y = 0
        # 保存结果
        temp_reuslt += temp_Y
        # 求一个浮点数的地板，就是求一个最接近它的整数 ceil向上取整
        if temp_reuslt < 0:
            temp_reuslt = 0
    # 结果修正
    modify = VM_TYPE_MODIFY3[vm_type]
    temp_reuslt = int(math.floor(temp_reuslt) + modify)
    if temp_reuslt < 0:
        temp_reuslt = 0
    result.append(temp_reuslt)
    return result


# 用例04  76.052
def predict_model14(his_data, date_range_size, vm_type):  # Holt-Winters法
    '''
    预测方案 十四 Holt-Winters
    :param his_data: 真实的历史数据出现次数表
    :param date_range_size: 需要预测的长度
    :param k:跨度天数
    :return: 返回结果
    77.325 alpha=1.8
    '''
    # 历史天数
    Y = copy.deepcopy(his_data['value'])

    k = 1
    temp_reuslt = 0.0
    result = []

    # 衰减值 0.24
    alpha = 0.243
    # 趋势
    beta = 0.000
    # 季节
    gamma = 0.21
    # 季度周期长度
    s = 7

    l_t = []
    b_t = []
    s_t = []

    # 初始trend
    pre_b_t = 0.0
    # 初始化level
    pre_l_t = 0.0
    # 初始化seasonal
    pre_s_t = 0.0

    # 初始化第一天的季动
    l_t.append(pre_l_t)
    b_t.append(pre_b_t)
    s_t.append(pre_s_t)

    # 在首部填充一位数据初始
    Y.insert(0, 0.0)

    # 用历史记录训练初始化参数
    for t in range(1, len(Y)):  # 当前是t时刻
        # 参数
        if (t - s) < len(s_t):  # 初始季动可能越界,越界则用上一个填充
            l_t.append(alpha * (Y[t] - s_t[t - 1]) + (1 - alpha) * (l_t[t - 1] + b_t[t - 1]))
        else:
            l_t.append(alpha * (Y[t] - s_t[t - s]) + (1 - alpha) * (l_t[t - 1] + b_t[t - 1]))

        b_t.append(beta * (l_t[t] - l_t[t - 1]) + (1 - beta) * b_t[t - 1])

        if (t - s) < len(s_t):  # 初始季动可能越界,越界则用上一个填充
            s_t.append(gamma * (Y[t] - l_t[t]) + (1 - gamma) * s_t[t - 1])
        else:
            s_t.append(gamma * (Y[t] - l_t[t]) + (1 - gamma) * s_t[t - s])

    t = len(l_t) - 1

    # 预测要预测的时间k为相隔多少天,相连预测数据相隔k=1
    for h in range(k, date_range_size + k):
        # 追加到历史表中
        # temp_Y = l_t[t] + h*b_t[t] + s_t[t - s+1+((h-1)%s)]

        temp_Y = l_t[t] + h * b_t[t] + s_t[t - s + h]
        # # 如果小于0 置为零
        # if temp_Y < 0:
        #     temp_Y = 0
        # 保存结果
        temp_reuslt += temp_Y
        # 求一个浮点数的地板，就是求一个最接近它的整数 ceil向上取整
        if temp_reuslt < 0:
            temp_reuslt = 0
    # 结果修正
    modify = VM_TYPE_MODIFY4[vm_type]
    temp_reuslt = int(math.floor(temp_reuslt) + modify)
    if temp_reuslt < 0:
        temp_reuslt = 0
    result.append(temp_reuslt)
    return result


###################对若干星期前同一天数据求平均######################
def predict_model15(his_data,  # 某种类型的虚拟机的历史数据
                    date_range_size, vm_type):  # 需要预测的长度



    n = 10  # 边长数10
    sigma = 0.5

    back_week = 1
    chis_data = copy.deepcopy(his_data['value'])
    cal_len = len(chis_data)

    result = []
    for rept in range(date_range_size):  # 预测天数范围
        day_avage = 0.0
        cot_week = 0
        for i in range(1, back_week + 1):
            index = i * 7
            if index <= cal_len:
                day_tmp = chis_data[-index] * n
                cot_day = n
                cot_week += 1
                for j in range(1, n):
                    tmp = (n - j) / 2.0
                    day_tmp += chis_data[-index + j] * tmp
                    cot_day += tmp
                    if index + j <= cal_len:
                        day_tmp += chis_data[-index - j] * tmp
                        cot_day += tmp
                    else:
                        continue
                day_avage += day_tmp / cot_day
            else:
                break
        # if cot_week != 0:
        #     day_avage = day_avage * 1.0 / cot_week  # 注意报错
        # if is_noise:
        #     noise = random.gauss(0, sigma)
        #     noise = math.fabs(noise)
        #     day_avage = int(math.ceil(day_avage + noise))
        # else:
        #     day_avage = int(math.ceil(day_avage))
        day_avage = int(math.ceil(day_avage))
        chis_data.append(day_avage)
        result.append(day_avage)

    return result


def predict_model16(his_data,  # 某种类型的虚拟机的历史数据
                    date_range_size, vm_type):  # 需要预测的长度

   

    n = 10  # 边长数 10
    sigma = 0.5

    beta = 2.0

    back_week = 1
    chis_data = copy.deepcopy(his_data['value'])
    cal_len = len(chis_data)

    result = []
    for rept in range(date_range_size):  # 预测天数范围
        day_avage = 0.0
        cot_week = 0
        for i in range(1, back_week + 1):
            index = i * 7
            if index <= cal_len:
                day_tmp = chis_data[-index] * n
                cot_day = n
                cot_week += 1
                for j in range(1, n):
                    tmp = (n - j) / beta
                    day_tmp += chis_data[-index + j] * tmp
                    cot_day += tmp
                    if index + j <= cal_len:
                        day_tmp += chis_data[-index - j] * tmp
                        cot_day += tmp
                    else:
                        continue
                day_avage += day_tmp / cot_day
            else:
                break
        if cot_week != 0:
            day_avage = day_avage * 1.0 / cot_week  # 注意报错
        if is_noise:
            noise = random.gauss(0, sigma)
            noise = math.fabs(noise)
            day_avage = int(math.ceil(day_avage + noise))
        else:
            day_avage = int(math.ceil(day_avage))

        day_avage = int(math.ceil(day_avage))
        chis_data.append(day_avage)
        result.append(day_avage)

    return result


def predict_model17(his_data,  # 某种类型的虚拟机的历史数据
                    date_range_size, vm_type):  # 需要预测的长度
    

    n = 2  # 边长数2  83.075
    sigma = 0.5

    back_week = 1  # 1
    chis_data = copy.deepcopy(his_data['value'])
    cal_len = len(chis_data)

    result = []
    for rept in range(date_range_size):  # 预测天数范围
        day_avage = 0.0
        cot_week = 0
        for i in range(1, back_week + 1):
            index = i * 7
            if index <= cal_len:
                day_tmp = chis_data[-index] * n
                cot_day = n
                cot_week += 1
                for j in range(1, n):
                    tmp = (n - j) / 2.0
                    day_tmp += chis_data[-index + j] * tmp
                    cot_day += tmp
                    if index + j <= cal_len:
                        day_tmp += chis_data[-index - j] * tmp
                        cot_day += tmp
                    else:
                        continue
                day_avage += day_tmp / cot_day
            else:
                break
        if cot_week != 0:
            day_avage = day_avage * 1.0 / cot_week  # 注意报错
        if is_noise:
            noise = random.gauss(0, sigma)
            noise = math.fabs(noise)
            day_avage = int(math.ceil(day_avage + noise))
        else:
            day_avage = int(math.ceil(day_avage))
        chis_data.append(day_avage)
        result.append(day_avage)

    return result


def predict_model18(his_data,  # 某种类型的虚拟机的历史数据
                    date_range_size, vm_type):  # 需要预测的长度

    

    n = 10  # 边长数10 83.11
    sigma = 0.5

    back_week = 1  # 1
    chis_data = copy.deepcopy(his_data['value'])
    cal_len = len(chis_data)

    result = []
    for rept in range(date_range_size):  # 预测天数范围
        day_avage = 0.0
        cot_week = 0
        for i in range(1, back_week + 1):
            index = i * 7
            if index <= cal_len:
                day_tmp = chis_data[-index] * n
                cot_day = n
                cot_week += 1
                for j in range(1, n):
                    tmp = (n - j) / 2.0
                    day_tmp += chis_data[-index + j] * tmp
                    cot_day += tmp
                    if index + j <= cal_len:
                        day_tmp += chis_data[-index - j] * tmp
                        cot_day += tmp
                    else:
                        continue
                day_avage += day_tmp / cot_day
            else:
                break
        if cot_week != 0:
            day_avage = day_avage * 1.0 / cot_week  # 注意报错
        if is_noise:
            noise = random.gauss(0, sigma)
            noise = math.fabs(noise)
            day_avage = int(math.ceil(day_avage + noise))
        else:
            day_avage = int(math.ceil(day_avage))
        chis_data.append(day_avage)
        result.append(day_avage)

    return result


#########################################LSTM#########################################
def predict_model19(caseInfo):  # 数据对象
    # 结果
    result = {}
    vm_types = caseInfo.vm_types
    # 获得训练集
    train_X = caseInfo.get_train_X()
    train_Y = caseInfo.get_train_Y()
    tic = t.time()
    # 最大最小训练集
    max_data = vt.find_max(caseInfo.train_X)
    min_data = vt.find_min(caseInfo.train_X)
    train_X = vt.normalize_uniform(train_X, max_data, min_data)

    max_data_Y = max_data[0:caseInfo.vm_types_size]
    min_data_Y = min_data[0:caseInfo.vm_types_size]

    input_size = len(train_X[0])
    label_size = caseInfo.vm_types_size
    output_size = label_size
    batch_size = 7
    num_epoch = 300
    learning_rate = 0.3
    num_batch_LB = int(math.floor(len(train_X) / batch_size))
    num_batch_UB = int(math.ceil(len(train_X) / batch_size))
    last_batch_size = len(train_X) % batch_size

    lstm = lstmcl.LSTM(input_size, output_size, batch_size)
    lstm.init_lstm(rand_type='gauss')

    cost = []

    view_epoch=50
    for i in range(num_epoch):
        cost_epoch = 0
        for j in range(num_batch_LB):
            X_batch = [[0] * input_size]
            X_batch.extend(train_X[j * batch_size:j * batch_size + batch_size])
            lstm.lstm_forward(X_batch)
            Label_batch = [[0] * label_size]
            Label_batch.extend(train_Y[j * batch_size:j * batch_size + batch_size])
            cost_batch_j = lstm.cost_LSE(Label_batch)
            cost_epoch = cost_epoch + cost_batch_j
            D = lstm.deriv_LSE(Label_batch)
            lstm.lstm_bptt(D, X_batch, learning_rate, 'Adam')
            lstm.reset()
        if num_batch_LB != num_batch_UB:
            lstm.set_batch_size(last_batch_size)
            X_batch_last = [[0] * input_size]
            X_batch_last.extend(train_X[len(train_X) - last_batch_size:len(train_X)])
            lstm.lstm_forward(X_batch_last)
            Label_batch_last = [[0] * label_size]
            Label_batch_last.extend(train_Y[len(train_Y) - last_batch_size:len(train_Y)])
            cost_batch_last = lstm.cost_LSE(Label_batch_last)
            cost_epoch = cost_epoch + cost_batch_last
            D = lstm.deriv_LSE(Label_batch_last)
            lstm.lstm_bptt(D, X_batch_last, learning_rate, 'Adam')
            lstm.reset()
        cost_epoch = cost_epoch / len(train_X)
        if i % view_epoch == 0:
            print('%d.epoch: cost -->%f' % (i / view_epoch, cost_epoch))
        cost.append(cost_epoch)

    # plt.figure()
    # plt.plot(cost)
    # plt.xlabel("number of epoch")
    # plt.ylabel("cost over one epoch")
    # plt.title("learning curve with learning rate: " + str(learning_rate))
    # plt.show()
    # X_pred = [[0]*input_size]
    # X_pred.extend(Xn)
    # lstm.set_batch_size(len(Xn))
    # lstm.lstm_forward(X_pred)
    # h = lstm.get_h()
    # pred = vt.ceil(vt.denormalize(h,mean,std))

    for j in range(num_batch_LB):
        X_batch = [[0] * input_size]
        X_batch.extend(train_X[j * batch_size:j * batch_size + batch_size])
        lstm.lstm_forward(X_batch)

    if num_batch_LB != num_batch_UB:
        lstm.set_batch_size(last_batch_size)
        X_batch_last = [[0] * input_size]
        X_batch_last.extend(train_X[len(train_X) - last_batch_size:len(train_X)])
        lstm.lstm_forward(X_batch_last)

    # lstm.set_batch_size(num_days_pred)
    h_prev = lstm.get_h()[-1]
    c_prev = lstm.get_c()[-1]

    # lstm.set_batch_size(num_days_pred)
    predict_result = vt.ceil(vt.denormalize_uniform([h_prev], max_data_Y, min_data_Y))

    st = datetime.datetime.strptime(caseInfo.data_range[0], '%Y-%m-%d %H:%M:%S')
    et = datetime.datetime.strptime(caseInfo.data_range[1], '%Y-%m-%d %H:%M:%S')
    td = datetime.timedelta(hours=24)
    time_fears = []
    # 遍历时间
    while st < et:
        time_fears.append(caseInfo.get_time_feature(st))
        st += td
    # 正规化
    time_fears = vt.normalize_uniform(time_fears, max_data[caseInfo.vm_types_size:], min_data[caseInfo.vm_types_size:])

    # 遍历时间
    for i in range(len(time_fears)):
        time_X = []
        time_X.extend(h_prev)
        time_X.extend(time_fears[i])
        c_prev, h_prev = lstm.predict_t(c_prev, h_prev, time_X)
        predict_result.extend(vt.trunc(vt.denormalize_uniform([h_prev], max_data, min_data)))

    # print("\n")
    # # print("Historical records:")
    # # count_his = vt.count([a[0:num_flavor] for a in tr_set])
    # # print(count_his)
    # #
    # # print("\n")
    # print("Predictions:")
    # for i in range(1,len(predict_result)):
    #     print(predict_result[i])
    count_pred = vt.count(predict_result[1:])
    count_pred_list = vt.count_list(predict_result[1:])
    # print("number of each flavor:")
    # # print(count_pred_list)
    # print(count_pred)
    # print ('\n')
    # print("Real data:")
    # test_list = caseInfo.get_test_list()
    # print(test_list)
    # print ('\n')
    # toc = t.time()
    # print("time for training LSTM: " + str(toc - tic))
    # print ('\n')
    result = dict(zip(caseInfo.vm_types, count_pred_list))
    # print ('result:')
    # print(result)
    # print ('\n')

    return result


#########################################LSTM#########################################
# 选择预测方案
model_used_func=predict_model10
# 按样例选择方案

model1_used_func = predict_model11

model2_used_func = predict_model12

model3_used_func = predict_model13

model4_used_func = predict_model14

# 指数平均
model9_used_func = predict_model9

model6_used_func = predict_model6
model8_used_func = predict_model8
model5_used_func = predict_model5
model10_used_func = predict_model10
################霍尔特线性趋势法#########################

model21_used_func = predict_model15
model22_used_func = predict_model16
model23_used_func = predict_model17
model24_used_func = predict_model18
#######################################################


# 预测7
model7_used_func = predict_model7

# 间隔短的方案
short_gap_used_func = predict_model12

long_gap_used_func = predict_model7

#########################################
lstm_model_used_func = predict_model19
