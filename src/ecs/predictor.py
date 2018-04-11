# -*- coding: utf-8 -*-

import CaseProcess
import packing_processer
import prediction_processer
import copy

from ParamInfo import VM_TYPE_DIRT, VM_PARAM

global res_use_pro
global vm_size
global vm
global pm_size
global pm
global try_result
global other_res_use_pro
global threshold

threshold = 90
other_res_use_pro = 0
res_use_pro = 0
vm_size = 0
vm = []
pm_size = 0
pm = []

try_result = {}

#使用深度学习模型
is_deeplearing=1

def predict_vm(ecs_lines, input_lines):
    '''
    :param ecs_lines:训练数据list<strLine>
    :param input_lines:类型要求list<strLine>
    :return:预测结果
    '''
    result = []
    if ecs_lines is None:
        print 'ecs information is none'
        return result
    if input_lines is None:
        print 'input file information is none'
        return result
    # 生成训练对象 Step 02
    caseInfo = CaseProcess.CaseInfo(input_lines, ecs_lines)

    # 使用RNN进行预测
    # predict_result = train_RNN(caseInfo)

    # 预测数据 Step 03
    if is_deeplearing==1:
        predict_result = prediction_processer.predict_deeplearning(caseInfo)
    else:
        predict_result = prediction_processer.predict_all(caseInfo)

    #############################################微调数量##################################
    global res_use_pro
    global vm_size
    global vm
    global pm_size
    global pm
    global try_result
    global other_res_use_pro
    vm_size, vm, pm_size, pm, res_use_pro, other_res_use_pro = packing_processer.pack_all(caseInfo, predict_result)
    pading_que = []

    # 搜索优先级
    if caseInfo.opt_target == 'CPU':
        pading_que = [1.0, 2.0, 4.0]
    else:
        pading_que = [4.0, 2.0, 1.0]

    try_result = copy.deepcopy(predict_result)
    end_vm_pos = 0
    # 找到第一个非0位[1,15]
    for vm_type in range(len(VM_TYPE_DIRT) - 1, -1, -1):
        if try_result.has_key(VM_TYPE_DIRT[vm_type]) and try_result[VM_TYPE_DIRT[vm_type]] > 0:  # 键值对存在
            end_vm_pos = vm_type
            break
    for que in range(3):
        # 在有数量的区间内填充[1,8]
        for vm_type in range(end_vm_pos, -1, -1):
            if try_result.has_key(VM_TYPE_DIRT[vm_type]) and try_result[VM_TYPE_DIRT[vm_type]] >= 0 and \
                    VM_PARAM[VM_TYPE_DIRT[vm_type]][2] == pading_que[que]:  # 键值对存在
                # 找到非0的,最大,虚拟机
                try_result_modify(try_result, caseInfo, -1, VM_TYPE_DIRT[vm_type])
                try_result_modify(try_result, caseInfo, 1, VM_TYPE_DIRT[vm_type])


    vm_size, vm, pm_size, pm, res_use_pro, other_res_use_pro = packing_processer.pack_all(caseInfo, try_result)
    print('MAX_USE_PRO=%.2f,MAX_OTHER_PRO=%f' % (res_use_pro,other_res_use_pro))

    #############################################微调数量##################################
    result = result_to_list(vm_size, vm, pm_size, pm)
    print(result)
    return result


def try_result_modify(predict_result, caseInfo, try_value, vm_type):
    global other_res_use_pro
    global res_use_pro
    global vm_size
    global vm
    global pm_size
    global pm
    global try_result
    try_predict = copy.deepcopy(predict_result)
    try_predict[vm_type][0] = try_predict[vm_type][0] + try_value
    if try_predict[vm_type][0] < 0:  # 小于0没有意义
        return
    try_vm_size, try_vm, try_pm_size, try_pm, try_res_use_pro, try_other_res_use_pro = packing_processer.pack_all(
        caseInfo, try_predict)
    if try_res_use_pro > res_use_pro and try_pm_size <= pm_size:  # 如果结果优,物理机数量相等或者 【更小,利用率更高 】保存最优结果
        vm_size, vm, pm_size, pm, res_use_pro, other_res_use_pro = try_vm_size, try_vm, try_pm_size, try_pm, try_res_use_pro, try_other_res_use_pro
        try_result = try_predict
        # 继续深度搜索
        try_result_modify(try_predict, caseInfo, try_value, vm_type)
    elif try_res_use_pro == res_use_pro and try_other_res_use_pro > other_res_use_pro:  # 如果没有当前的好,则返回
        vm_size, vm, pm_size, pm, res_use_pro, other_res_use_pro = try_vm_size, try_vm, try_pm_size, try_pm, try_res_use_pro, try_other_res_use_pro
        try_result = try_predict
        # 继续深度搜索
        try_result_modify(try_predict, caseInfo, try_value, vm_type)
    else:
        return


def result_to_list(vm_size, vm, pm_size, pm):
    '''
    由预测和分配生成结果
    vm：{vm_type:cot...}
    pm[{vm_type:cot,vm_type2:cot2...}...]
    '''
    end_str = ''
    result = []
    result.append(str(vm_size) + end_str)
    for index in vm.keys():
        item = vm[index]
        tmp = index + ' ' + str(item) + end_str
        result.append(tmp)

    result.append(end_str)
    # TODO
    result.append(str(pm_size) + end_str)
    for pm_id in range(len(pm)):
        tmp = str(pm_id + 1)
        pmone = pm[pm_id]
        if len(pmone.keys()) == 0:
            continue
        for index in pmone.keys():
            item = pmone[index]
            tmp += ' ' + index + ' ' + str(item)
        tmp += end_str
        result.append(tmp)
    return result
