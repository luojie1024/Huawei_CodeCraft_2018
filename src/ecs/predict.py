# -*- coding: utf-8 -*-

import DataObj
import predict_utils
import copy
import math
from const_map import VM_TYPE_DIRT, VM_PARAM, VM_CPU_QU, VM_MEM_QU
import packing_utils_v2

global res_use
global vm_size
global vm
global local_pm_size
global pm
global try_result
global threshold
global vm_map
global pm_name
global origin_pm_size
global local_optimal_result
global global_optimal_result
global local_res_use
global global_res_use
global local_c_m
global global_c_m

local_res_use = 0.0
global_res_use = 0.0

origin_pm_size = 0

local_c_m = 0.25
global_c_m = 0.25
pm_name = []

vm_map = {}
threshold = 90
res_use = 0
vm_size = 0
vm = []
local_pm_size = 0
pm = []

local_optimal_result = {}
global_optimal_result = {}
try_result = {}
#
is_parameter_search = False
# 使用深度学习模型
is_deeplearing = False
use_smooth = True
use_search_maximum = True
use_search_u_m_maximum = False


def predict_vm(ecs_lines, input_lines, input_test_file_array=None):
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
    dataObj = DataObj.DataObj(input_lines, ecs_lines, input_test_file_array=input_test_file_array)

    # 使用RNN进行预测
    # predict_result = train_RNN(dataObj)

    predict_result = predict_utils.predict_all(dataObj)

    # 参数搜索
    if is_parameter_search == False:

        # 预测数据 Step 03
        if is_deeplearing:
            predict_result = predict_utils.predict_deeplearning(dataObj)
        else:
            predict_result = predict_utils.predict_all(dataObj)
    #############################################参数搜索##################################
    else:
        parameter = {"alpha": 0, "beta": 0, "gamma": 0}
        max_score = 0.0
        for alpha in range(1, 100, 2):
            for beta in range(1, 100, 2):
                for gamma in range(1, 100, 2):
                    predict_result = predict_utils.predict_predict_parameter(dataObj, alpha / 100.0, beta / 100.0,
                                                                             gamma / 100.0)
                    # 评估函数
                    score = evaluation_parameter(dataObj, predict_result)
                    if score > max_score:
                        parameter['alpha'] = alpha
                        parameter['beta'] = beta
                        parameter['gamma'] = gamma
                        max_score = score
            print('%d:alpha=%d,beta=%d,gamma=%d,max_score=%f\n' % (
                alpha, parameter['alpha'], parameter['beta'], parameter['gamma'], max_score))
        print('max_paremeter:')

        print(parameter)
        print('max_score:%f' % max_score)
    #############################################微调数量##################################
    global global_optimal_result
    global local_c_m
    global vm_map
    global local_pm_size
    global origin_pm_size
    global local_c_m
    global global_c_m
    global local_res_use
    origin_use_rate = 0.0

    # 虚拟机表
    vm_map = dict(zip(dataObj.vm_types, [0] * dataObj.vm_types_size))

    vm_size, vm, pm_size, pm, pm_name, local_res_use, pm_free = packing_utils_v2.pack_api(dataObj,
                                                                                          predict_result,
                                                                                          local_c_m)
    local_pm_size = pm_size
    origin_use_rate = local_res_use
    origin_pm_size = local_pm_size
    #############################################use_pm_average##################################
    # if use_search_u_m_maximum:
    #     search_u_m_maximum(dataObj, predict_result)
    #     vm_size, vm, pm_size, pm, pm_name, res_use, pm_free = packing_utils_v2.pack_api(dataObj,
    #                                                                                     try_result)
    #############################################use_pm_average##################################

    #############################################use_search_maximum##################################
    if use_search_maximum:
        search_maximum_way1(dataObj, predict_result)
        vm_size, vm, pm_size, pm, pm_name, res_use, pm_free = packing_utils_v2.pack_api(dataObj,
                                                                                        global_optimal_result,
                                                                                        global_c_m)
        print('use_search_use_rate=%.5f%%\n' % (res_use))
    #############################################use_search_maximum##################################

    #############################################use_smooth##################################
    if use_smooth:
        vm_size, vm, pm_size, pm, add_cpu, add_mem = result_smooth(vm_size, vm, pm_size, pm, dataObj, pm_free)
        print('use_smooth_use_rate=%.5f%% + cpu=+%d mem=+%d  \n' % (res_use, add_cpu, add_mem))

    #############################################use_smooth##################################

    print('origin_use_rate=%.5f%%\n' % (origin_use_rate))
    # # 评估函数
    # if is_parameter_search == False:
    #     evaluation(dataObj, predict_result)

    result = result_to_list(vm_size, vm, pm_size, pm, pm_name, dataObj.pm_type_name)
    print(result)
    return result


def search_maximum_way1(dataObj, predict_result):
    global vm_size
    global vm
    global pm_name
    global pm
    global try_result
    global vm_map
    global local_optimal_result
    global local_res_use
    global local_pm_size
    global local_c_m
    global global_c_m
    global global_res_use
    global global_optimal_result

    # 寻找最优CM比例
    target_c_m = [0.25, 0.5, 1, None]
    # 初始化 暂存跟节点情况
    pm_size = local_pm_size
    res_use = local_res_use

    for i in range(len(target_c_m)):

        try_vm_size, try_vm, try_pm_size, try_pm, try_pm_name, try_res_use, _ = packing_utils_v2.pack_api(dataObj,
                                                                                                          predict_result,
                                                                                                          target_c_m[i])
        # if (try_res_use) > (res_use) and try_pm_size <= pm_size:
        if (try_res_use) > (res_use) and try_pm_size <= pm_size:
            local_c_m = target_c_m[i]
            _, _, pm_size, _, _, res_use = try_vm_size, try_vm, try_pm_size, try_pm, try_pm_name, try_res_use
    # 赋值最优的大小
    pm_size += 2
    local_pm_size = pm_size
    local_res_use = res_use
    global_res_use = local_res_use
    global_c_m = local_c_m

    Weight_que1 = [4.0, 2.0, 1.0]
    Weight_que2 = [2.0, 1.0, 2.0]
    Weight_que3 = [1.0, 4.0, 2.0]
    # 搜索优先级
    # if dataObj.opt_target == 'CPU':
    #     pading_que = [1.0, 2.0, 4.0]
    # else:
    #     pading_que = [4.0, 2.0, 1.0]

    # 根据数量初始化队列
    local_optimal_result = copy.deepcopy(predict_result)
    global_optimal_result = copy.deepcopy(predict_result)
    # 遍历所有放置队列 九个方向寻找最优解 TODO 可能超时 容易陷入局部最优
    pading_que = [1.0, 1.0, 1.0]
    for i in range(len(Weight_que1)):
        pading_que[0] = Weight_que1[i]
        for j in range(len(Weight_que2)):
            pading_que[1] = Weight_que2[j]
            for k in range(len(Weight_que3)):
                pading_que[2] = Weight_que3[k]

                # 根据数量初始化队列
                pre_copy = copy.deepcopy(local_optimal_result)

                end_vm_pos = 0
                # 找到第一个非0位[1,15]
                for vm_type_index in range(len(VM_TYPE_DIRT) - 1, -1, -1):
                    if pre_copy.has_key(VM_TYPE_DIRT[vm_type_index]) and pre_copy[
                        VM_TYPE_DIRT[vm_type_index]] > 0:  # 键值对存在
                        end_vm_pos = vm_type_index
                        break
                for que in range(3):
                    # 在有数量的区间内填充[1,8]
                    for vm_type in range(end_vm_pos, -1, -1):
                        if pre_copy.has_key(VM_TYPE_DIRT[vm_type]) and VM_PARAM[VM_TYPE_DIRT[vm_type]][2] == pading_que[
                            que]:  # 键值对存在,C/M比相等
                            if pre_copy[VM_TYPE_DIRT[vm_type]][0] > 0:
                                result_modify1(pre_copy, dataObj, 1, VM_TYPE_DIRT[vm_type], vm_map)
                                result_modify1(pre_copy, dataObj, -1, VM_TYPE_DIRT[vm_type], vm_map)
                            else:
                                # 找到非0的,最大,虚拟机
                                result_modify1(pre_copy, dataObj, 1, VM_TYPE_DIRT[vm_type], vm_map)
                # 保存最优解
                if local_res_use > global_res_use:
                    global_optimal_result = local_optimal_result
                    global_res_use = local_res_use
                    global_c_m = local_c_m
                # 初始化局部最优解
                local_optimal_result = copy.deepcopy(predict_result)
                # 初始化使用率
                local_res_use = res_use
                # 回滚主机数量
                local_pm_size = pm_size


def result_modify1(predict_result, dataObj, try_value, vm_type, try_vm_map):
    '''
    :param predict_result: 虚拟机预测结果 贪心搜索局部优解
    :param dataObj: 训练集信息
    :param try_value: 尝试值
    :param vm_type: 虚拟机类型
    :return:
    '''
    global res_use
    global vm_size
    global vm
    global local_pm_size
    global pm
    global try_result
    global vm_map
    global pm_name
    global local_c_m
    global global_c_m
    global global_optimal_result
    global local_optimal_result
    global local_res_use
    global global_res_use

    try_predict = copy.deepcopy(predict_result)
    try_vm_map = copy.deepcopy(vm_map)
    try_predict[vm_type][0] = try_predict[vm_type][0] + try_value
    if try_predict[vm_type][0] < 0:  # 小于0没有意义
        return
    # try_vm_size, try_vm, try_pm_size, try_pm, try_pm_name, try_res_use, _ = packing_utils_v2.pack_api(
    #     dataObj, try_predict)

    # 遍历各种不同优化比例
    target_c_m = [0.25, 0.5, 1, None]
    for i in range(len(target_c_m)):
        try_vm_size, try_vm, try_pm_size, try_pm, try_pm_name, try_res_use, _ = packing_utils_v2.pack_api(dataObj,
                                                                                                          try_predict,
                                                                                                          target_c_m[i])
        if (try_res_use) > (local_res_use) and try_pm_size <= local_pm_size:  # 如果结果优,物理机数量相等或者 【更小,利用率更高 】保存最优结果
            _, _, local_pm_size, _, _, local_res_use = try_vm_size, try_vm, try_pm_size, try_pm, try_pm_name, try_res_use
            local_optimal_result = try_predict
            try_vm_map[vm_type] += try_value
            vm_map = try_vm_map
            local_c_m = target_c_m[i]
            # 继续深度搜索
            result_modify1(try_predict, dataObj, try_value, vm_type, try_vm_map)
        else:
            continue
    return


def result_smooth(vm_size, vm, pm_size, pm, dataObj, pm_free):
    '''
    平滑填充结果集
    :param vm:虚拟机列表
    :param pm_size:虚拟机数量
    :param pm:物理机列表
    :param dataObj:数据对象
    :return:
    '''
    vm_types = dataObj.vm_types
    res_use_pro = 0.0
    other_res_use_pro = 0.0
    VM_QUE = []
    free_cpu = 0.0
    free_mem = 0.0
    # 初始化填充队列
    # if dataObj.opt_target == 'CPU':
    #     VM_QUE = VM_CPU_QU
    #     res_use_pro = dataObj.CPU * pm
    #     other_res_use_pro = dataObj.MEM * pm
    # else:
    #     VM_QUE = VM_MEM_QU
    #     res_use_pro = dataObj.MEM * pm
    #     other_res_use_pro = dataObj.CPU * pm

    VM_QUE = VM_CPU_QU
    add_cpu = 0
    add_mem = 0
    epoch = 1
    # 遍历物理机
    for i in range(pm_size - 2, pm_size):
        M_C = 0.0
        # 进行多轮赋值,防止漏空
        for e in range(epoch):  # CPU 内存均有空间
            if pm_free[i][0] and pm_free[i][1]:
                # 计算占比
                is_all_pack = 0
                M_C = computer_MC(pm_free[i])
                while (M_C >= 1 and pm_free[i][0] and pm_free[i][1] and is_all_pack < 3):  # CPU 内存均有空间
                    # 3轮不同比例的检索
                    for vm_type_index in range(len(VM_PARAM) - 1, -1, -1):
                        # 比例匹配,并且是属于预测列表的最大资源虚拟机
                        if VM_PARAM[VM_TYPE_DIRT[vm_type_index]][2] == M_C and (
                                VM_TYPE_DIRT[vm_type_index] in vm_types):
                            # CPU 内存均有空间放入该虚拟机
                            if VM_PARAM[VM_TYPE_DIRT[vm_type_index]][0] <= pm_free[i][0] and \
                                    VM_PARAM[VM_TYPE_DIRT[vm_type_index]][1] <= pm_free[i][1]:
                                # 虚拟机数量增加
                                vm_size += 1

                                if isContainKey(vm, VM_TYPE_DIRT[vm_type_index]):
                                    # 列表中数量添加
                                    vm[VM_TYPE_DIRT[vm_type_index]] += 1
                                else:
                                    vm[VM_TYPE_DIRT[vm_type_index]] = 1
                                # 物理机列表中添加
                                if isContainKey(pm[i], VM_TYPE_DIRT[vm_type_index]):
                                    pm[i][VM_TYPE_DIRT[vm_type_index]] += 1
                                else:
                                    pm[i][VM_TYPE_DIRT[vm_type_index]] = 1
                                # 剪切空闲空间数
                                pm_free[i][0] = pm_free[i][0] - VM_PARAM[VM_TYPE_DIRT[vm_type_index]][0]
                                pm_free[i][1] = pm_free[i][1] - VM_PARAM[VM_TYPE_DIRT[vm_type_index]][1]
                                add_cpu += VM_PARAM[VM_TYPE_DIRT[vm_type_index]][0]
                                add_mem += VM_PARAM[VM_TYPE_DIRT[vm_type_index]][1]
                                # 无空闲资源,则跳出循环
                                if pm_free[i][0] == 0 or pm_free[i][1] == 0:
                                    break
                # 只进行三轮检索
                is_all_pack += 1
    return vm_size, vm, pm_size, pm, add_cpu, add_mem


def search_u_m_maximum(dataObj, predict_result):
    '''
    搜寻最优的资源CM优化比例
    :param dataObj:
    :param predict_result:
    :return:
    '''
    global res_use
    global vm_size
    global vm
    global local_pm_size
    global pm_name
    global pm
    global try_result
    global vm_map
    pass


# 检查dict中是否存在key
def isContainKey(dic, key):
    return key in dic


def evaluation_parameter(dataObj, vm):
    diff, score = diff_dic(dataObj.test_vm_count, vm)
    return score


def evaluation(dataObj, vm):
    print('train count:\n')
    print(dataObj.train_vm_count)
    print('\n')

    print('test count:\n')
    print(dataObj.test_vm_count)
    print('\n')

    print('predict count:\n')
    print(vm)
    print('\n')

    # diff, score = diff_dic(dataObj.test_vm_count, vm)
    diff, score = diff_dic(dataObj.test_vm_count, vm)

    print('diff count:\n')
    print(diff)
    print('\n')

    print('score count:\n')
    print(score)
    print('\n')

    return score


def diff_dic(test, predict):
    '''
    计算差
    :param test:测试集
    :param predict:预测数据
    :return: 差值
    '''
    diff = {}
    score = 1.0
    n = 0.0
    temp_diff = 0.0
    temp_y = 0.0
    temp_y_hot = 0.0
    keys = test.keys()
    for key in keys:
        if isContainKey(predict, key):
            diff[key] = predict[key][0] - test[key]
            temp_y_hot += (predict[key][0] ** 2)
        else:
            diff[key] = abs(test[key])
        temp_diff += (diff[key] ** 2)
        temp_y += (test[key] ** 2)
        n += 1.0

    score -= math.sqrt(temp_diff / n) / (math.sqrt(temp_y / n) + math.sqrt(temp_y_hot / n))

    return diff, score


def computer_MC(CM_free):
    # 计算内存/CPU占比
    M_C = CM_free[1] / CM_free[0]
    if M_C >= 4:
        M_C = 4.0
    elif M_C >= 2:
        M_C = 2.0
    else:
        M_C = 1.0
    return M_C


# def init_que(caseInfo):
#     vm_que=[]
#     for vm_type in caseInfo.vm_types:
#
#     return vm_que

def result_to_list(vm_size, vm, pm_size, pm, pm_name, pm_type_name):
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

    pm_datas = {}
    # 物理机信息处理
    for i in range(len(pm)):
        # 虚拟机类型key
        name = pm_name[i]
        if isContainKey(pm_datas, name):
            # 添加数据
            pm_datas[name].append(pm[i])
        else:
            pm_datas[name] = []
            pm_datas[name].append(pm[i])
    end_str_sum = 2
    for name in pm_type_name:
        # 如果当前类型物理机请求量为0
        if not isContainKey(pm_datas, name):
            result.append(name + ' ' + str(0) + end_str)
            if end_str_sum > 0:
                result.append(end_str)
                end_str_sum -= 1
            continue
        else:
            result.append(name + ' ' + str(len(pm_datas[name])) + end_str)

        # 每一种物理机
        pm_data = pm_datas[name]
        for pm_id in range(len(pm_datas[name])):
            tmp = name + '-' + str(pm_id + 1)
            # 每一行物理机
            pm_item = pm_data[pm_id]
            # 一个都没有
            if len(pm_item.keys()) == 0:
                continue

            for index in pm_item.keys():
                item = pm_item[index]
                tmp += ' ' + index + ' ' + str(item)
            tmp += end_str
            result.append(tmp)

        if end_str_sum > 0:
            result.append(end_str)
            end_str_sum -= 1

    # result.append(end_str)
    # result.append(pm_type_name[1] + ' ' + str(0))
    # result.append(end_str)
    # result.append(pm_type_name[2] + ' ' + str(0))
    return result
