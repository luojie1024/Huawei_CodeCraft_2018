# -*- coding: utf-8 -*-

from datetime import datetime

import predict_model
import predict_model2

import BPNN

# 需要预测的天数2
range_size1 = 7
# 需要预测的天数3
range_size2 = 7

# 类型1的虚拟机阈值
preliminar1_1size = 3
preliminar1_2_size = 5

# 类型2的虚拟机阈值
preliminar2_1_size = 3
preliminar2_2_size = 5

# 历史列表类型 1 2 滑动均值 3真实值
vmtype_avage_v = 3


def predict_deeplearning(dataObj):
    '''
    输入为DataObj对象,使用深度学习模型
    :param dataObj:
    :return: 预测结果
    '''
    result = {}
    # 使用LSTM
    predict_func = predict_model.lstm_model_used_func
    # 预测
    result = predict_func(dataObj)
    # 返回结果
    return result


def predict_all(dataObj):
    result = {}
    vm_types = dataObj.vm_types

    global vmtype_avage_v

    # result=predict_func(dataObj)

    # 预测天数[7] [7]

    # predict_func = short_gap_predict_func  # 76.68
    # if dataObj.date_range_size==7:
    #     predict_func = long_gap_predict_func# 78
    # else:
    #     predict_func=short_gap_predict_func#76.68

    # 时间间隔  连续gap_time=1
    gap_time = dataObj.gap_time

    start_time = datetime.strptime(dataObj.data_range[0], "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(dataObj.data_range[1], "%Y-%m-%d %H:%M:%S")
    # 2016 4 5月份
    pos_time1 = datetime.strptime('2016-04-08 00:00:00', "%Y-%m-%d %H:%M:%S")
    pos_time2 = datetime.strptime('2016-04-15 00:00:00', "%Y-%m-%d %H:%M:%S")
    # pos_time3 = datetime.strptime('2016-04-16 00:00:00', "%Y-%m-%d %H:%M:%S")
    # pos_time4 = datetime.strptime('2016-04-18 00:00:00', "%Y-%m-%d %H:%M:%S")
    # pos_time5 = datetime.strptime('2016-04-20 00:00:00', "%Y-%m-%d %H:%M:%S")

    # 虚拟机类型数
    vm_type_size = dataObj.vm_types_size

    # 需要预测的天数
    data_size = dataObj.date_range_size
    '''
    #每个等级的难度主要根据预测的时间长短以及预测的虚拟机规格数量两个指标来区分。 ,按照虚拟机规格数量||预测时间区分 (初赛按虚拟机规格区分)
    #样例1  2016-04-08  预测的天数=7  虚拟机类型==3
    #样例2  2016-04-08  预测的天数=7  虚拟机类型==5
    #样例3  2016-04-15  预测的天数=7  虚拟机类型==3
    #样例4  2016-04-15  预测的天数=7  虚拟机类型==5
    # L1 天数(0, 7] 间隔=1
    # L2 天数(7,14] 间隔(1,8]
    '''
    #################################################Holt-Winters##################################################
    # #训练数据多少天
    # print((end_time-start_time).days)

    # 均值处理
    # if data_size <= 7:
    #     predict_func = predict_model2.model1_used_func  # model1_used_func 75.091
    #     # 5x5滤波
    #     vmtype_avage_v = 6
    #
    # # Holt-Winters
    # elif data_size <= 14:  # 样例2 L2  2016-04-08  预测天数 7 虚拟机类型5 (3,5]
    #     predict_func = predict_model2.model2_used_func  # model2_used_func	77.092
    #     # 3x3滤波
    #     vmtype_avage_v = 6
    # elif  data_size <= 21:  # 样例2 L2  2016-04-08  预测天数 7 虚拟机类型5 (3,5]
    #     predict_func = predict_model.model22_used_func  # model2_used_func	77.092
    # elif  data_size <= 28:  # 样例2 L2  2016-04-08  预测天数 7 虚拟机类型5 (3,5]
    #     predict_func = predict_model.model22_used_func  # model2_used_func	77.092

    #################################################星期前同一天数据求平均##################################################

    # if end_time == pos_time1 and data_size == range_size1 and vm_type_size <= preliminar1_1size:  # 样例1  L1 2016-04-08  预测天数 7 虚拟机类型3
    #     predict_func = predict_model.model21_used_func  # model1_used_func 75.091
    #     vmtype_avage_v = 1
    # elif end_time == pos_time1 and data_size == range_size1 and vm_type_size > preliminar1_1size and vm_type_size <=preliminar1_2_size:  # 样例2 L2  2016-04-08  预测天数 7 虚拟机类型5 (3,5]
    #     predict_func = predict_model.model22_used_func  # model2_used_func	77.092
    #     vmtype_avage_v = 1
    # elif end_time == pos_time2 and data_size == range_size2 and vm_type_size <= preliminar2_1_size:  # 样例3 L1   2016-04-15 预测天数7
    #     predict_func = predict_model.model23_used_func  # model3_used_func  77.32
    #     # predict_func = predict_model.model23_used_func  # 78.712
    #     vmtype_avage_v = 1
    # elif end_time == pos_time2 and data_size == range_size2 and vm_type_size > preliminar2_1_size and vm_type_size<=preliminar2_2_size:  # 样例4  L2  2016-04-15 预测天数7  虚拟机类型5 (3,5]
    #     predict_func = predict_model.model24_used_func  # model4_used_func 77.156
    #     vmtype_avage_v = 1
    #################################################星期前同一天数据求平均##################################################

    #################################################MAX-SCORE##################################################

    # if end_time == pos_time1 and data_size == range_size1 and vm_type_size <= preliminar1_1size:  # 样例1  L1 2016-04-08  预测天数 7 虚拟机类型3
    #     predict_func = predict_model.model1_used_func  # model1_used_func 75.091
    # elif end_time == pos_time1 and data_size == range_size1 and vm_type_size > preliminar1_1size and vm_type_size <=preliminar1_2_size:  # 样例2 L2  2016-04-08  预测天数 7 虚拟机类型5 (3,5]
    #     predict_func = predict_model.model22_used_func  # model2_used_func	77.092
    # elif end_time == pos_time2 and data_size == range_size2 and vm_type_size <= preliminar2_1_size:  # 样例3 L1   2016-04-15 预测天数7
    #     predict_func = predict_model.model3_used_func  # model3_used_func  77.32
    #     # predict_func = predict_model.model23_used_func  # 78.712
    # elif end_time == pos_time2 and data_size == range_size2 and vm_type_size > preliminar2_1_size and vm_type_size<=preliminar2_2_size:  # 样例4  L2  2016-04-15 预测天数7  虚拟机类型5 (3,5]
    #     predict_func = predict_model.model4_used_func  # model4_used_func 77.156
    #################################################MAX-SCORE##################################################


    if gap_time>1:
        predict_func = predict_model2.model1_used_func

    # 3x3填充方案
    vmtype_avage_v = 6

    # vmtype_avage_v = 6

    # predict_func = predict_model.model9_used_func
    # vmtype_avage_v=3

    for vmtype in vm_types:
        result[vmtype] = predict(vmtype, dataObj, predict_func)
    # for vmtype in vm_types:
    #     result[vmtype] = predict_BPNN(vmtype, dataObj, predict_func)

    return result


def predict(vm_type,  # 虚拟机类型
            dataObj,  # 案例信息对象
            prodict_function=None,  # 时间序列预测
            ):
    return prodict_function(dataObj.get_data_list(vm_type, -1, vmtype_avage_v),
                            dataObj, vm_type)

# def predict_BPNN(vm_type,  # 虚拟机类型
#                 dataObj,  # 案例信息对象
#                 prodict_function=None,  # 时间序列预测):
#     return prodict_function(dataObj.get_data_list(vm_type, -1,vmtype_avage_v),
#                                 dataObj.date_range_size, vm_type)


#  test here
# return [1,2,3,4]
