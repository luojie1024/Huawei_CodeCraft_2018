# -*- coding: utf-8 -*-
import copy
import math
import random

from ParamInfo import *

#加入随机数
is_noise=0

def predict_model1(his_data,  # 某种类型的虚拟机的历史数据
                   date_range_size,vm_type):  # 需要预测的长度

    '''
    预测方案一,使用MV模型预测，最近n天 
    历史长度为n个粒度时间，权重设定暂定，
    his_data:['time':[时间标签],'value':[值]]
    '''
    n = 7  # 历史长度
    # 权重，从最近到最久，长度为n
    ws = [0.45, 0.25,
          0.15, 0.08,
          0.04, 0.02,
          0.01]
    chis_data = copy.deepcopy(his_data['value'])
    result = []
    for rept in range(date_range_size):
        cal_len = len(chis_data)
        tmpn = 0
        predict = 0.0
        for i in range(cal_len - 1, -1, -1):
            predict += chis_data[i] * ws[tmpn]
            tmpn += 1
            if tmpn == n: break
        predict = int(math.floor(predict))
        chis_data.append(predict)
        result.append(predict)
    return result


def predict_model2(his_data,  # 某种类型的虚拟机的历史数据
                   date_range_size,vm_type):  # 需要预测的长度

    '''
    预测方案二,使用SMV 模型预测， 最近n天
    历史长度为n个粒度时间，权重设定暂定，
    his_data:['time':[时间标签],'value':[值]]
    '''
    n = 14  # 历史长度
    # 权重，从最近到最久，长度为n

    chis_data = copy.deepcopy(his_data['value'])
    result = []
    for rept in range(date_range_size):
        cal_len = len(chis_data)
        tmpn = 0
        predict = 0.0
        for i in range(cal_len - 1, -1, -1):
            predict += chis_data[i]
            tmpn += 1
            if tmpn == n: break
        # TODO 2018 3 21 Fix ZeroDivisionError: float division by zero
        if tmpn != 0:
            predict = int(math.ceil(predict * 1.0 / tmpn))
        else:
            predict = 0
        chis_data.append(predict)
        result.append(predict)
    return result


def predict_model3(his_data,  # 某种类型的虚拟机的历史数据
                   date_range_size,vm_type):  # 需要预测的长度

    '''
    预测方案3,使用MV 模型预测，最近n天
    历史长度为n个粒度时间，使用倒数权重，
    his_data:['time':[时间标签],'value':[值]]
    '''
    n = 10  # 历史长度
    # 权重，从最近到最久，长度为n
    ws = []
    ws_sum = 0.0
    for i in range(n):
        # tmp = 1.0/(1.0+i)
        tmp = math.exp(-1.0 * i)
        ws_sum += tmp
        ws.append(tmp)
    for i in range(n):
        ws[i] = ws[i] / ws_sum

    chis_data = copy.deepcopy(his_data['value'])
    result = []
    for rept in range(date_range_size):
        cal_len = len(chis_data)
        tmpn = 0
        predict = 0.0
        for i in range(cal_len - 1, -1, -1):
            predict += chis_data[i] * ws[tmpn]
            tmpn += 1
            if tmpn == n: break
        predict = int(math.ceil(predict))
        chis_data.append(predict)
        result.append(predict)
    return result


def predict_model4(his_data,  # 某种类型的虚拟机的历史数据
                   date_range_size,vm_type):  # 需要预测的长度

    '''
    预测方案四,使用SMV 模型预测，添加正态随机噪声  最近n天
    历史长度为n个粒度时间，权重设定暂定，
    his_data:['time':[时间标签],'value':[值]]
    '''
    n = 13  # 历史长度
    sigma = 0.4

    chis_data = copy.deepcopy(his_data['value'])
    result = []
    for rept in range(date_range_size):
        cal_len = len(chis_data)
        tmpn = 0
        predict = 0.0
        for i in range(cal_len - 1, -1, -1):
            predict += chis_data[i]
            tmpn += 1
            if tmpn == n: break
        noise = random.gauss(0, sigma)
        predict = int(math.ceil(predict * 1.0 / tmpn + noise))
        chis_data.append(predict)
        result.append(predict)
    return result


def front_out(w, b, x):
    suma = 0.0
    for i in range(len(w)):
        suma += w[i] * x[i]
    return suma + b[0]


def change(w, b, x, py_y, lr):
    for i in range(len(w)):
        w[i] -= py_y * x[i] * lr
    b[0] -= lr * py_y


def predict_model5(his_data,  # 某种类型的虚拟机的历史数据
                   date_range_size,vm_type):  # 需要预测的长度

    '''
    预测方案5,进行一次差分，然后使用MA处理   失败
    历史长度为n个粒度时间，使用倒数权重，
    his_data:['time':[时间标签],'value':[值]]
    '''
    n = 10  # 历史长度
    sigma = 0.1
    # 权重，从最近到最久，长度为n
    lr = 0.01

    origin_data = his_data['value']
    chis_data = copy.deepcopy(origin_data)

    # 差分,计算平均
    diff = []
    avag = []
    for i in range(1, len(chis_data)):
        b = origin_data[i]
        a = origin_data[i - 1]
        diff.append(b - a)
        avag.append((a + b) / 2.0)
    avag.append(origin_data[-1])

    print 'diff', diff
    print 'avag', avag
    result = []

    w = []
    for i in range(n):
        w.append(random.gauss(0, 0.1))
    b = [random.gauss(0, 0.1)]

    cal_len = len(chis_data)
    for i in range(n, cal_len):
        x = chis_data[i - n:i]
        py = front_out(w, b, x)
        y = chis_data[i]
        #         print py,y
        #         print w,b
        change(w, b, x, py - y, lr)

    for rept in range(date_range_size):
        cal_len = len(chis_data)
        tmpn = 0
        predict = 0.0
        x = chis_data[cal_len - n:]
        y = []
        predict = front_out(w, b, x)
        # change(w,b,x,predict-y,lr)

        noise = random.gauss(0, sigma)
        predict = int(math.ceil(predict * 1.0 + noise))
        diff.append(predict)
        predict = predict + chis_data[-1]
        # chis_data.append(predict+chis_data[-1])
        result.append(predict)
    return result


def predict_model6(his_data,  # 某种类型的虚拟机的历史数据
                   date_range_size,vm_type):  # 需要预测的长度

    '''
    预测方案六,使用SMV模型预测，添加正态随机噪声 
    先将数据进行一次平均处理，然后采用平均预测 
    历史长度为n个粒度时间，权重设定暂定，
    his_data:['time':[时间标签],'value':[值]]
    '''
    # 如果当前主机类型在当前数据集未出现
    if (his_data['time'] == 0):
        return [0]

    n = 7  # 历史长度
    sigma = 0.1

    n_layer1 = 1
    chis_data = copy.deepcopy(his_data['value'])
    cal_len = len(chis_data)
    avag = []
    tmp = 0.0
    last = 0.0
    for i in range(n_layer1):
        last = chis_data[i] * 1.0 / n_layer1
        tmp += last
    avag.append(tmp)
    for i in range(n_layer1, cal_len):
        tmp -= last
        last = chis_data[i] * 1.0 / n_layer1
        tmp += last
        avag.append(tmp)

    result = []
    for rept in range(date_range_size):
        cal_len = len(avag)
        tmpn = 0
        predict = 0.0
        for i in range(cal_len - 1, -1, -1):
            predict += avag[i]
            tmpn += 1
            if tmpn == n: break
        noise = random.gauss(0, sigma)
        predict = int(math.ceil(predict * 1.0 / tmpn + noise))
        avag.append(predict)
        result.append(predict)
    return result


def predict_model7(his_data,  # 某种类型的虚拟机的历史数据
                   date_range_size,vm_type):  # 需要预测的长度

    '''
    预测方案七,对若干星期前同一天数据求平均
    his_data:['time':[时间标签],'value':[值]]
    '''

    n = 3  # 边长数
    sigma = 0.5

    back_week = 2
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
        noise = random.gauss(0, sigma)
        noise = math.fabs(noise)
        day_avage = int(math.ceil(day_avage + noise))
        chis_data.append(day_avage)
        result.append(day_avage)

    return result

def predict_model8(his_data,  # 某种类型的虚拟机的历史数据
                   date_range_size,vm_type):  # 需要预测的长度

    '''
    预测方案七,对若干星期前同一天数据求平均
    his_data:['time':[时间标签],'value':[值]]
    '''

    n = 2  # 边长数
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
        if cot_week != 0:
            day_avage = day_avage * 1.0 / cot_week  # 注意报错
        noise = random.gauss(0, sigma)
        noise = math.fabs(noise)
        day_avage = int(math.ceil(day_avage + noise))
        # day_avage = int(math.ceil(day_avage))
        chis_data.append(day_avage)
        result.append(day_avage)

    return result

def predict_model9(his_data, date_range_size,vm_type):  # 简单滑动平均法
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


def predict_model10(his_data, date_range_size,vm_type):  # 霍尔特线性趋势法
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
    result.append(int(math.floor(temp_reuslt)+noise))
    return result


################Holt-Winters#########################
#用例01  76.68  小于三种类型
def predict_model11(his_data, date_range_size, vm_type):  # Holt-Winters法
    '''
    预测方案 十一 Holt-Winters
    :param his_data: 真实的历史数据出现次数表
    :param date_range_size: 需要预测的长度
    :param k:跨度天数
    :return: 返回结果
    '''
    Y = copy.deepcopy(his_data['value'])


    k=1

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

    #在首部填充一位数据初始
    Y.insert(0,0.0)

    # 用历史记录训练初始化参数
    for t in range(1, len(Y)): # 当前是t时刻
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

    t=len(l_t)-1

    # 预测要预测的时间k为相隔多少天,相连预测数据相隔k=1
    for h in range(k,date_range_size+k):
            # 追加到历史表中
        # temp_Y = l_t[t] + h*b_t[t] + s_t[t - s+1+((h-1)%s)]

        temp_Y = l_t[t] + h * b_t[t] + s_t[t - s + h]
        # # 如果小于0 置为零
        # if temp_Y < 0:
        #     temp_Y = 0
        # 保存结果
        temp_reuslt += temp_Y
        # 求一个浮点数的地板，就是求一个最接近它的整数 ceil向上取整
        if temp_reuslt<0:
            temp_reuslt=0

    # 结果修正
    modify = VM_TYPE_MODIFY1[vm_type]
    temp_reuslt = int(math.floor(temp_reuslt) + modify)
    if temp_reuslt < 0:
        temp_reuslt = 0
    result.append(temp_reuslt)
    return result



#用例02  76.147
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
    k=1

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

    #在首部填充一位数据初始
    Y.insert(0,0.0)

    # 用历史记录训练初始化参数
    for t in range(1, len(Y)): # 当前是t时刻
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

    t=len(l_t)-1

    # 预测要预测的时间k为相隔多少天,相连预测数据相隔k=1
    for h in range(k,date_range_size+k):
            # 追加到历史表中
        # temp_Y = l_t[t] + h*b_t[t] + s_t[t - s+1+((h-1)%s)]

        temp_Y = l_t[t] + h * b_t[t] + s_t[t - s + h]
        # # 如果小于0 置为零
        # if temp_Y < 0:
        #     temp_Y = 0
        # 保存结果
        temp_reuslt += temp_Y
        # 求一个浮点数的地板，就是求一个最接近它的整数 ceil向上取整
        if temp_reuslt<0:
            temp_reuslt=0

    # 结果修正
    modify = VM_TYPE_MODIFY2[vm_type]
    temp_reuslt = int(math.floor(temp_reuslt) + modify)
    if temp_reuslt < 0:
        temp_reuslt = 0
    result.append(temp_reuslt)
    return result


#用例03 小于三种类型
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
    k=1

    temp_reuslt = 0.0
    result = []

    # 衰减值0.186
    alpha = 0.186
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

    #在首部填充一位数据初始
    Y.insert(0,0.0)

    # 用历史记录训练初始化参数
    for t in range(1, len(Y)): # 当前是t时刻
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

    t=len(l_t)-1

    # 预测要预测的时间k为相隔多少天,相连预测数据相隔k=1
    for h in range(k,date_range_size+k):
            # 追加到历史表中
        # temp_Y = l_t[t] + h*b_t[t] + s_t[t - s+1+((h-1)%s)]

        temp_Y = l_t[t] + h * b_t[t] + s_t[t - s + h]
        # # 如果小于0 置为零
        # if temp_Y < 0:
        #     temp_Y = 0
        # 保存结果
        temp_reuslt += temp_Y
        # 求一个浮点数的地板，就是求一个最接近它的整数 ceil向上取整
        if temp_reuslt<0:
            temp_reuslt=0
    #结果修正
    modify=VM_TYPE_MODIFY3[vm_type]
    temp_reuslt=int(math.floor(temp_reuslt)+modify)
    if temp_reuslt < 0:
        temp_reuslt = 0
    result.append(temp_reuslt)
    return result



#用例04  76.052
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

    k=1
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

    #在首部填充一位数据初始
    Y.insert(0,0.0)

    # 用历史记录训练初始化参数
    for t in range(1, len(Y)): # 当前是t时刻
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

    t=len(l_t)-1

    # 预测要预测的时间k为相隔多少天,相连预测数据相隔k=1
    for h in range(k,date_range_size+k):
            # 追加到历史表中
        # temp_Y = l_t[t] + h*b_t[t] + s_t[t - s+1+((h-1)%s)]

        temp_Y = l_t[t] + h * b_t[t] + s_t[t - s + h]
        # # 如果小于0 置为零
        # if temp_Y < 0:
        #     temp_Y = 0
        # 保存结果
        temp_reuslt += temp_Y
        # 求一个浮点数的地板，就是求一个最接近它的整数 ceil向上取整
        if temp_reuslt<0:
            temp_reuslt=0
    result.append(int(math.floor(temp_reuslt)))
    return result



###################对若干星期前同一天数据求平均######################
def predict_model15(his_data,  # 某种类型的虚拟机的历史数据
                   date_range_size,vm_type):  # 需要预测的长度

    '''
       预测方案七,对若干星期前同一天数据求平均
       his_data:['time':[时间标签],'value':[值]]
       '''

    n = 10 # 边长数10
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

def predict_model16(his_data,  # 某种类型的虚拟机的历史数据
                   date_range_size,vm_type):  # 需要预测的长度

    '''
       预测方案七,对若干星期前同一天数据求平均
       his_data:['time':[时间标签],'value':[值]]
       '''

    n = 10  # 边长数 10
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

def predict_model17(his_data,  # 某种类型的虚拟机的历史数据
                   date_range_size,vm_type):  # 需要预测的长度
    '''
       预测方案七,对若干星期前同一天数据求平均
       his_data:['time':[时间标签],'value':[值]]
       '''

    n = 2  # 边长数2  83.075
    sigma = 0.5

    back_week = 1  #1
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
                   date_range_size,vm_type):  # 需要预测的长度

    '''
       预测方案七,对若干星期前同一天数据求平均
       his_data:['time':[时间标签],'value':[值]]
       '''

    n = 10  # 边长数2
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


#########################################
# 选择预测方案

#按样例选择方案

model1_used_func = predict_model11

model2_used_func = predict_model12

model3_used_func = predict_model13

model4_used_func = predict_model14

#指数平均
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


#预测7
model7_used_func=predict_model7

#间隔短的方案
short_gap_used_func = predict_model12

long_gap_used_func = predict_model7

#########################################
