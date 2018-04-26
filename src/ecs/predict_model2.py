# -*- coding: utf-8 -*-
import copy
import math
import random
import martix_utils as vt
import LSTM as lstmcl
import time as t
import datetime
import math

from BPNN import BPNeuralNetwork
from const_map import *

# 加入随机数
is_noise = True


def predict_model1(his_data, dataObj, vm_type):  # 但模型,不区分样例 80.032
    # 无noise
    # 需要预测的天数
    date_range_size = dataObj.date_range_size
    sigma = 0.5
    # 获取放大权重
    # count_weight=dataObj.get_count_weight(vm_type)

    n = 3
    # 放大系数
    enlarge = 1.35  # 1.35  80.032 .
    beta = 2.0
    back_week = 1
    chis_data = copy.deepcopy(his_data['value'])
    cal_len = len(chis_data)

    result = []
    temp_result = 0
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

        # 系数放大,修正高斯效果
        day_avage = day_avage * enlarge

        # 加入噪声
        if is_noise:
            noise = random.gauss(0, sigma)
            noise = math.fabs(noise)
            day_avage = int(math.floor(day_avage + noise))
        chis_data.append(day_avage)
        temp_result += day_avage
    result.append(temp_result)

    return result


def predict_model2(his_data, dataObj, vm_type):  # 按间隔时间区分样例  84.625
    # 无noise
    # 需要预测的天数
    date_range_size = dataObj.date_range_size
    sigma = 0.5
    # 获取放大权重
    # count_weight=dataObj.get_count_weight(vm_type)

    n = 3
    if dataObj.gap_time == 1:  # 连续
        enlarge = 1
    elif dataObj.gap_time <= 8:  # 间隔7天
        enlarge = 1.49  # 1.49  84.625
    else:  # 间隔15天
        enlarge = 1.49
    beta = 2.0
    back_week = 1
    chis_data = copy.deepcopy(his_data['value'])
    cal_len = len(chis_data)

    result = []
    temp_result = 0
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

        # 系数放大,修正高斯效果
        day_avage = day_avage * enlarge

        # 加入噪声
        if is_noise:
            noise = random.gauss(0, sigma)
            noise = math.fabs(noise)
            day_avage = int(math.floor(day_avage + noise))
        chis_data.append(day_avage)
        temp_result += day_avage
    result.append(temp_result)

    return result


def predict_model3(his_data, dataObj, vm_type):  # 霍尔特线性趋势法 {'alpha': 5, 'beta': 55, 'gamma': 1}
    '''
    预测方案 2 Holt-Winters
    :param his_data: 真实的历史数据出现次数表
    :param date_range_size: 需要预测的长度
    :param k:跨度天数
    :return: 返回结果
    '''
    # 获取权重
    weight = PREDICT_MODEL2_WEIGHTS[vm_type]
    # 预测天数
    date_range_size = dataObj.date_range_size
    # 历史映射
    Y = copy.deepcopy(his_data['value'])

    # 时间间隔跨度
    k = dataObj.gap_time

    temp_reuslt = 0.0
    result = []
    # 2.65

    # 衰减值 220
    # alpha = weight['alpha']
    alpha = 0.05
    # 趋势
    # beta = weight['beta']
    beta = 0.55
    # 季节 0.215
    # gamma = weight['gamma']
    gamma = 0.01
    # 季度周期长度
    s = weight['s']
    s = 7

    l_t = []
    b_t = []
    s_t = []

    # 初始trend
    pre_b_t = 1.0
    # 初始化level
    pre_l_t = 1.0
    # 初始化seasonal
    pre_s_t = 1.0

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
            l_t.append(alpha * (Y[t] / s_t[t - 1]) + (1 - alpha) * (l_t[t - 1] + b_t[t - 1]))
        else:
            l_t.append(alpha * (Y[t] / s_t[t - s]) + (1 - alpha) * (l_t[t - 1] + b_t[t - 1]))

        b_t.append(beta * (l_t[t] - l_t[t - 1]) + (1 - beta) * b_t[t - 1])

        if (t - s) < len(s_t):  # 初始季动可能越界,越界则用上一个填充
            s_t.append(gamma * (Y[t] / l_t[t]) + (1 - gamma) * s_t[t - 1])
        else:
            s_t.append(gamma * (Y[t] / l_t[t]) + (1 - gamma) * s_t[t - s])

    t = len(l_t) - 1

    # 预测要预测的时间k为相隔多少天,相连预测数据相隔k=1
    for h in range(k, date_range_size + k):
        # 追加到历史表中
        temp_Y = (l_t[t] + h * b_t[t]) * s_t[t - s + 1 + ((h - 1) % s)]

        # temp_Y = l_t[t] + h * b_t[t] + s_t[t - s + h]
        # # 如果小于0 置为零
        # if temp_Y < 0:
        #     temp_Y = 0
        # 保存结果
        temp_reuslt += temp_Y
        # 求一个浮点数的地板，就是求一个最接近它的整数 ceil向上取整
        if temp_reuslt < 0:
            temp_reuslt = 0

    enlarge = weight['enlarge']

    enlarge = enlarge * dataObj.get_count_weight(vm_type, temp_reuslt, dataObj.date_range_size)

    temp_reuslt = temp_reuslt * (2 + enlarge)

    # 结果修正
    temp_reuslt = int(math.floor(temp_reuslt))
    if temp_reuslt < 0:
        temp_reuslt = 0
    result.append(temp_reuslt)
    return result


def predict_model4(his_data, dataObj, vm_type):  # 霍尔特线性趋势法
    '''
    预测方案 2 Holt-Winters
    :param his_data: 真实的历史数据出现次数表
    :param date_range_size: 需要预测的长度
    :param k:跨度天数
    :return: 返回结果
    '''
    # 获取权重
    weight = PREDICT_MODEL2_WEIGHTS[vm_type]
    # 预测天数
    date_range_size = dataObj.date_range_size
    # 历史映射
    Y = copy.deepcopy(his_data['value'])

    # 时间间隔跨度
    k = dataObj.gap_time

    temp_reuslt = 0.0
    result = []

    # 衰减值 220
    alpha = weight['alpha']
    # alpha = 0.06
    # 趋势
    beta = weight['beta']
    # beta = 0.6
    # 季节 0.215
    gamma = weight['gamma']
    # gamma = 0.08
    # 季度周期长度
    s = weight['s']
    s = 7

    l_t = []
    b_t = []
    s_t = []

    # 初始trend
    pre_b_t = 1.0
    # 初始化level
    pre_l_t = 1.0
    # 初始化seasonal
    pre_s_t = 1.0

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
            l_t.append(alpha * (Y[t] / s_t[t - 1]) + (1 - alpha) * (l_t[t - 1] + b_t[t - 1]))
        else:
            l_t.append(alpha * (Y[t] / s_t[t - s]) + (1 - alpha) * (l_t[t - 1] + b_t[t - 1]))

        b_t.append(beta * (l_t[t] - l_t[t - 1]) + (1 - beta) * b_t[t - 1])

        if (t - s) < len(s_t):  # 初始季动可能越界,越界则用上一个填充
            s_t.append(gamma * (Y[t] / l_t[t]) + (1 - gamma) * s_t[t - 1])
        else:
            s_t.append(gamma * (Y[t] / l_t[t]) + (1 - gamma) * s_t[t - s])

    t = len(l_t) - 1

    # 预测要预测的时间k为相隔多少天,相连预测数据相隔k=1
    for h in range(k, date_range_size + k):
        # 追加到历史表中
        temp_Y = (l_t[t] + h * b_t[t]) * s_t[t - s + 1 + ((h - 1) % s)]

        # temp_Y = l_t[t] + h * b_t[t] + s_t[t - s + h]
        # # 如果小于0 置为零
        # if temp_Y < 0:
        #     temp_Y = 0
        # 保存结果
        temp_reuslt += temp_Y
        # 求一个浮点数的地板，就是求一个最接近它的整数 ceil向上取整
        if temp_reuslt < 0:
            temp_reuslt = 0

    enlarge = weight['enlarge']

    # enlarge = enlarge * dataObj.get_count_weight(vm_type,temp_reuslt,dataObj.date_range_size)

    temp_reuslt = temp_reuslt * enlarge

    # 结果修正
    temp_reuslt = int(math.floor(temp_reuslt))
    if temp_reuslt < 0:
        temp_reuslt = 0
    result.append(temp_reuslt)
    return result


def predict_model5(his_data, dataObj, vm_type):  # 样例+类型区分 86.7
    # 无noise
    # 需要预测的天数
    date_range_size = dataObj.date_range_size
    sigma = 0.5
    # 获取放大权重
    # count_weight=dataObj.get_count_weight(vm_type)

    if dataObj.gap_time == 1:  # 预测时间7天,间隔1天(连续),5种类
        weight = PREDICT_MODEL1_WEIGHTS[vm_type]
    elif dataObj.gap_time <= 8:  # 预测时间14天,间隔7天(=8),8种类预测类型
        weight = PREDICT_MODEL21_WEIGHTS[vm_type]
    else:
        weight = PREDICT_MODEL21_WEIGHTS[vm_type]
    # if dataObj.gap_time == 1:  # 无间隔 7天预测
    #     weight = PREDICT_MODEL1_WEIGHTS[vm_type]
    # elif dataObj.gap_time > 1 and dataObj.gap_time <= 8:  # 间隔7天
    #     weight = PREDICT_MODEL21_WEIGHTS[vm_type]
    # elif dataObj.gap_time > 8 :  # 间隔7天
    #     weight = PREDICT_MODEL21_WEIGHTS[vm_type]

    n = weight['n']
    # 放大系数
    enlarge = weight['enlarge']
    beta = weight['beta']
    back_week = weight['back_week']
    chis_data = copy.deepcopy(his_data['value'])
    cal_len = len(chis_data)

    result = []
    temp_result = 0
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

        # 系数放大,修正高斯效果
        day_avage = day_avage * enlarge

        # 加入噪声
        if is_noise:
            noise = random.gauss(0, sigma)
            noise = math.fabs(noise)
            day_avage = int(math.floor(day_avage + noise))
        chis_data.append(day_avage)
        temp_result += day_avage
    result.append(temp_result)

    return result


def predict_model6(his_data, dataObj, vm_type):  # 霍尔特线性趋势法
    '''
    预测方案 2 Holt-Winters
    :param his_data: 真实的历史数据出现次数表
    :param date_range_size: 需要预测的长度
    :param k:跨度天数
    :return: 返回结果
    '''
    # 获取权重
    weight = PREDICT_MODEL2_WEIGHTS[vm_type]
    # 预测天数
    date_range_size = dataObj.date_range_size
    # 历史映射
    Y = copy.deepcopy(his_data['value'])

    # 时间间隔跨度
    k = dataObj.gap_time

    temp_reuslt = 0.0
    result = []
    # 2.65

    # 衰减值 220
    alpha = weight['alpha']
    alpha = 0.22
    # 趋势
    beta = weight['beta']
    beta = 0.7
    # 季节 0.215
    gamma = weight['gamma']
    gamma = 0.215
    # 季度周期长度
    s = weight['s']
    # s = 7

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
        temp_Y = l_t[t] + h * b_t[t] + s_t[t - s + 1 + ((h - 1) % s)]

        # temp_Y = l_t[t] + h * b_t[t] + s_t[t - s + h]
        # # 如果小于0 置为零
        # if temp_Y < 0:
        #     temp_Y = 0
        # 保存结果
        temp_reuslt += temp_Y
        # 求一个浮点数的地板，就是求一个最接近它的整数 ceil向上取整
        if temp_reuslt < 0:
            temp_reuslt = 0

    # enlarge = weight['enlarge']
    #
    # enlarge = enlarge * dataObj.get_count_weight(vm_type,temp_reuslt,dataObj.date_range_size)
    #
    # temp_reuslt = temp_reuslt * (1.5+enlarge)

    # 结果修正
    temp_reuslt = int(math.floor(temp_reuslt))
    if temp_reuslt < 0:
        temp_reuslt = 0
    result.append(temp_reuslt)
    return result

def predict_model7(his_data, dataObj, vm_type):
 # 无noise
    # 需要预测的天数
    date_range_size = dataObj.date_range_size
    sigma = 0.5
    # 获取放大权重
    # count_weight=dataObj.get_count_weight(vm_type)

    # if dataObj.gap_time == 1:  # 预测时间7天,间隔1天(连续),5种类
    #     weight = PREDICT_MODEL1_WEIGHTS[vm_type]
    # elif dataObj.date_range_size > 7 and dataObj.date_range_size <= 14 and dataObj.gap_time > 8 and dataObj.gap_time <= 8:  # 预测时间14天,间隔7天(=8),8种类预测类型
    #     weight = PREDICT_MODEL21_WEIGHTS[vm_type]
    # else:
    #     weight = PREDICT_MODEL21_WEIGHTS[vm_type]

    if dataObj.gap_time > 8 and dataObj.gap_time <= 8:  # 预测时间14天,间隔7天(=8),8种类预测类型
        weight = PREDICT_MODEL21_WEIGHTS[vm_type]

    n = weight['n']
    # 放大系数
    enlarge = weight['enlarge']
    beta = weight['beta']
    back_week = weight['back_week']
    chis_data = copy.deepcopy(his_data['value'])
    cal_len = len(chis_data)

    result = []
    temp_result = 0
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

        # 系数放大,修正高斯效果
        day_avage = day_avage * enlarge

        # 加入噪声
        if is_noise:
            noise = random.gauss(0, sigma)
            noise = math.fabs(noise)
            day_avage = int(math.floor(day_avage + noise))
        chis_data.append(day_avage)
        temp_result += day_avage
    result.append(temp_result)

    return result

model1_used_func = predict_model1

model2_used_func = predict_model2

model3_used_func = predict_model3

model4_used_func = predict_model4

model5_used_func = predict_model5

model7_used_func = predict_model7