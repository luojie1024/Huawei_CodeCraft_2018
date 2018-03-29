# -*- coding: utf-8 -*-

'''
    预测模型，输入CaseInfo对象，
    输出在预测期内各个虚拟机类型的请求数量
    
'''

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

    #预测天数[8,8]
    if caseInfo.date_range_size>=5:
        predict_func=long_gap_predict_func#78
    else:
        predict_func=short_gap_predict_func#76

    for vmtype in vm_types:
        result[vmtype] = predict_one(vmtype,caseInfo,predict_func)

    return result


def predict_one(vm_type,# 虚拟机类型
                caseInfo,# 案例信息对象
                prodict_function=None,# 时间序列预测
                ):

    return prodict_function(caseInfo.get_his_data_by_vmtype_avage_v3(vm_type,-1),
                            caseInfo.date_range_size,caseInfo.gap_time)
    '''
    训练并预测一种虚拟机的类型，返回为
    一个[v1,v2,v3....]预测结果数组
    '''
    
    #  test here
    # return [1,2,3,4]







