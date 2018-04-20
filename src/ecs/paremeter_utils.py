# -*- coding: utf-8 -*-
'''
# @Time    : 18-4-19 下午7:34
# @Author  : luojie
# @File    : paremeter_utils.py
# @Desc    : 参数搜索搜索工具
'''

import copy


def write_result(array, outpuFilePath):
    with open(outpuFilePath, 'w') as output_file:
        for item in array:
            output_file.write("%s\n" % item)


outpuFilePath = 'result/'

VM_PARAM_F = {
    'flavor1': [1, 1, 1.0],
    'flavor2': [1, 2, 2.0],
    'flavor3': [1, 4, 4.0],

    'flavor4': [2, 2, 1.0],
    'flavor5': [2, 4, 2.0],
    'flavor6': [2, 8, 4.0],

    'flavor7': [4, 4, 1.0],
    'flavor8': [4, 8, 2.0],
    'flavor9': [4, 16, 4.0],

    'flavor10': [8, 8, 1.0],
    'flavor11': [8, 16, 2.0],
    'flavor12': [8, 32, 4.0],

    'flavor13': [16, 16, 1.0],
    'flavor14': [16, 32, 2.0],
    'flavor15': [16, 64, 4.0],

    'flavor16': [32, 32, 1.0],
    'flavor17': [32, 64, 2.0],
    'flavor18': [32, 128, 4.0],
}

INDEX_MAP={
    'flavor1': 1,
    'flavor2': 2,
    'flavor3': 3,

    'flavor4': 4,
    'flavor5': 5,
    'flavor6': 6,

    'flavor7': 7,
    'flavor8': 8,
    'flavor9': 9,

    'flavor10': 10,
    'flavor11': 11,
    'flavor12': 12,

    'flavor13': 13,
    'flavor14': 14,
    'flavor15': 15,

    'flavor16': 16,
    'flavor17': 17,
    'flavor18': 18,
}

VM_PARAM = {
    'flavor1': [1, 1, 1.0],
    'flavor2': [1, 2, 2.0],
    'flavor3': [1, 4, 4.0],

    'flavor4': [2, 2, 1.0],
    'flavor5': [2, 4, 2.0],
    'flavor6': [2, 8, 4.0],

    'flavor7': [4, 4, 1.0],
    'flavor8': [4, 8, 2.0],
    'flavor9': [4, 16, 4.0],

    # 'flavor10': [8, 8, 1.0],
    # 'flavor11': [8, 16, 2.0],
    # 'flavor12': [8, 32, 4.0],
    #
    # 'flavor13': [16, 16, 1.0],
    # 'flavor14': [16, 32, 2.0],
    # 'flavor15': [16, 64, 4.0],
    #
    # 'flavor16': [32, 32, 1.0],
    # 'flavor17': [32, 64, 2.0],
    'flavor18': [32, 128, 4.0],
}

VM_PM = {
    'General':
        {'CPU': 56, 'MEM': 128, 'HHD': 1200},
    'Large-Memory':
        {'CPU': 84, 'MEM': 256, 'HHD': 2400},
    'High-Performance':
        {'CPU': 112, 'MEM': 192, 'HHD': 3600}
}

PM = {'CPU': 84, 'MEM': 256, 'HHD': 2400}

path_list = {}

res_list = {}

result = []


def fun(free_res_list, path_list):
    # 遍历列表
    for key in VM_PARAM.keys():
        # 完美组合
        if (VM_PARAM[key][0] == free_res_list['CPU'] and VM_PARAM[key][1] == free_res_list['MEM']):
            path = add_vm(path_list, key)
            # 保存结果
            result.append(path)
        elif (VM_PARAM[key][0] < free_res_list['CPU'] and VM_PARAM[key][1] < free_res_list['MEM']):
            # 减去资源数
            free = delete_res(free_res_list, key)
            # 添加路径
            path = add_vm(path_list,key)
            # 继续搜索
            fun(free, path)
        else:
            continue


def delete_res(res_list, key):
    '''
    :param res_list:资源列表
    :param key: 减去资源
    :return: 返回修改后的资源
    '''
    free_res_list = copy.deepcopy(res_list)
    free_res_list['CPU'] -= VM_PARAM[key][0]
    free_res_list['MEM'] -= VM_PARAM[key][1]
    return free_res_list


def add_vm(path_list, key):
    path=copy.deepcopy(path_list)
    if isContainKey(path,key):
        path[key]+=1
    else:
        path[key]=1
    return path


# 检查dict中是否存在key
def isContainKey(dic, key):
    return key in dic


def result_unique(data):
    if len(data)==0:
        return []
    temp=data[0]
    unique_result=[]
    unique_result.append(temp)
    for i in range(1,len(data)):
        if cmp(temp,data[i]):
            temp=data[i]
            unique_result.append(copy.deepcopy(data[i]))
        else:
            continue

    return unique_result


for key in VM_PM.keys():
    PM = VM_PM[key]
    fun(PM, path_list)
    result.sort()
    print('%s result  size=%d' % (key, len(result)))
    result=result_unique(result)
    write_result(result, outpuFilePath + key +'vm18' +'.txt')
    print('%s result_unique size=%d' % (key, len(result)))
    print('\n end \n')
    result = []

print('\nend!\n')
