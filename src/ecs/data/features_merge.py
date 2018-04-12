# -*- coding: utf-8 -*-
import csv
import copy

from ParamInfo import VM_TYPE_DIRT

'''
# @Time    : 18-3-23 下午7:46
# @Author  : luojie
# @File    : features_merge.py
# @Desc    : 特征合并
'''
#读特征映射表
time_list={}
times_column=[]
values_column=[]
feartures_list=[]

def feature_merge(vm_types,his_data):
    '''
    :param caseInfo:
    :return: 特征集合  vmtype_list = {'time': [0],  # 时间标签
                                    'value': [0]}  # 统计值
    '''
    is_head=True
    with open('data/data_features.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        #读取某第一列时间,去掉文件头
        for row in reader:
            if is_head:
                is_head=False
                continue
            times_column.append(row[0])
            # value = [int(x) for x in row[1:]]
            #转字符串成数字
            value=map(int, row[1:])
            values_column.append(value)
        # times_column = [row for row in reader][1:]

    # with open('data/data_features.csv', 'rb') as csvfile:
    #     reader = csv.reader(csvfile)
    #     #读取某第后面列,去掉文件头
    #     # numbers = map(int, row[1:])
    #     values_column = [row[1:]for row in reader][1:]
    #以时间为键,特征为值的表值
    time_list = {times_column[i]: values_column[i] for i in range(len(times_column))}

    # time_list=dict(map(lambda x, y: [x, y], times_column, values_column))
    # time_list = dict(zip(times_column[0::1], [values_column[0::1]]))
    #检查长度
    # for time_item in time_list:
    #     print len(time_item)
    #需要预测的虚拟机种类
    # caseInfo.vm_types_size

    #遍历需要预测的flavors类型,进行特征map的建立
    for vmtype in vm_types:#his_data--> flavors1：{'2015-02-15 00:00:00': 1, '2015-02-10 00:00:00': 2}
        #虚拟机{时间：次数}字典{'2015-02-15 00:00:00': 1, '2015-02-10 00:00:00': 2}
        if his_data.has_key(vmtype):
            vm_time_count_lists=his_data[vmtype]
        else:
            continue
            print "not find his_data value"
        # 获取时间建表 ['2015-02-15 00:00:00','2015-02-10 00:00:00']
        vm_time_lists=vm_time_count_lists.keys()
        for index,vm_time_item in enumerate(vm_time_lists):
            #从特征表中获取特征值
            # print(time_list.get(vm_time_item))
            if time_list.has_key(vm_time_item):
                #深拷贝一个列表
                feartures_temp=copy.copy(time_list.get(vm_time_item))
            else:
                print "not find vm_time_item value"
            for type in VM_TYPE_DIRT:
                if type==vmtype:
                    feartures_temp.append(1)
                else:
                    feartures_temp.append(0)
            # y 从历史表中获取当前主机出现的次数次数
            feartures_temp.append(vm_time_count_lists.get(vm_time_item))
            #保存每一行信息 34列 x[0:33] y[-1]
            # print "feartures_temp len="
            # print len(feartures_temp)
            feartures_list.append(feartures_temp)
    #返回整个特征列表
    return feartures_list
# feature_merge()