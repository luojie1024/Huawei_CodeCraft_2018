# -*- coding: utf-8 -*-

'''
    预测模型，输入CaseInfo对象，
    输出在预测期内各个虚拟机类型的请求数量
    
'''
from datetime import datetime

import predict_model

long_gap_predict_func = predict_model.long_gap_used_func

short_gap_predict_func = predict_model.short_gap_used_func


def predict_all(caseInfo):
    '''
    输入为CaseInfo对象，
    返回一个结果对象，结构为{vm_type:[v1,v2,v3....]}
    数组长度为caseInfo,中date_range_size,代表各个时间粒度内，
    该虚拟机被请求数,
    注意：当前预测模型设置只适合各个虚拟机类型独立预测
    '''
    result = {}
    vm_types = caseInfo.vm_types

    # result=predict_func(caseInfo)

    # 预测天数[7] [7]

    # predict_func = short_gap_predict_func  # 76.68
    # if caseInfo.date_range_size==7:
    #     predict_func = long_gap_predict_func# 78
    # else:
    #     predict_func=short_gap_predict_func#76.68

    start_time = datetime.strptime(caseInfo.data_range[0], "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(caseInfo.data_range[1], "%Y-%m-%d %H:%M:%S")
    # 2016 4 5月份
    pos_time1 = datetime.strptime('2016-04-08 00:00:00', "%Y-%m-%d %H:%M:%S")
    pos_time2 = datetime.strptime('2016-04-15 00:00:00', "%Y-%m-%d %H:%M:%S")
    # pos_time3 = datetime.strptime('2016-04-16 00:00:00', "%Y-%m-%d %H:%M:%S")
    # pos_time4 = datetime.strptime('2016-04-18 00:00:00', "%Y-%m-%d %H:%M:%S")
    # pos_time5 = datetime.strptime('2016-04-20 00:00:00', "%Y-%m-%d %H:%M:%S")

    # 需要预测的天数1
    range_size1=9
    # 需要预测的天数2
    range_size2=8

    # 需要预测的天数2
    range_size3=6
    # 需要预测的天数3
    range_size4=7

    #需要预测的天数
    data_size=caseInfo.date_range_size
    '''
    #样例1  2016-04-08  预测的天数=9
    #样例2  2016-04-08  预测的天数=8   没有=8
    #样例3  2016-04-08  预测的天数=7   
    #样例4  2016-04-15  预测的天数=7
    
    '''
    if end_time == pos_time1 and data_size<range_size3:#样例1  2016-04-08  预测天数9 [0,7)
        predict_func = predict_model.model1_used_func  # 76.68

    elif end_time == pos_time1 and data_size==7:#样例1  2016-04-08  预测天数[8,9)
        predict_func = predict_model.model2_used_func  # 76.147

    elif end_time == pos_time1 and data_size==range_size3:#样例1  2016-04-08  预测天数 7  [preliminariesL1usecase01] [preliminariesL2usecase01]
        predict_func = predict_model.model3_used_func  # 76.147

    elif end_time == pos_time2 and data_size==range_size4:#样例2  2016-04-15 预测天数7  #	39.567
        predict_func = predict_model.model4_used_func  # 77.361


    # elif end_time >= pos_time3 and end_time < pos_time4 :
    #     predict_func = predict_model.model3_used_func # 76.94
    # elif end_time >= pos_time4 and end_time < pos_time5:
    #     predict_func = predict_model.model4_used_func # 76.052

    # model 1
    # predict_func = predict_model.model1_used_func  # 76.68
    # # model 2
    # predict_func = predict_model.model2_used_func  # 76.147
    # model 3
    # predict_func = predict_model.model3_used_func  # 76.94
    # model 4
    # predict_func = predict_model.model4_used_func  # 76.052

    for vmtype in vm_types:
        result[vmtype] = predict_one(vmtype, caseInfo, predict_func)

    return result


def predict_one(vm_type,  # 虚拟机类型
                caseInfo,  # 案例信息对象
                prodict_function=None,  # 时间序列预测
                ):
    return prodict_function(caseInfo.get_his_data_by_vmtype_avage_v3(vm_type, -1),
                            caseInfo.date_range_size, caseInfo.gap_time)
    '''
    训练并预测一种虚拟机的类型，返回为
    一个[v1,v2,v3....]预测结果数组
    '''

    #  test here
    # return [1,2,3,4]
