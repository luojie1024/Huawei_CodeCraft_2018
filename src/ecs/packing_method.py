# -*- coding: utf-8 -*-
import copy
from math import ceil

import packing_utils
from const_map import VM_TYPE_DIRT, VM_PARAM


def pack_model(vmPicker, serverObj, opt_target='CPU'):
    '''
    具体装配方案1,packing1,M/U权重分配
    '''
    # 获得放置顺序
    vm_orders = [[], # vm_types
                 []] # cot
    weightes = [1,2,4]
    cpu = [1,2,4,8,16]

    vm_cpu_size,vm_mem_size = vmPicker.origin_cpu_mem_sum()
    
    if vm_cpu_size == 0: return  # 无需装装配， 结束
    
    pw =vm_mem_size*1.0 / vm_cpu_size
    
    C = serverObj.server_info['CPU']# 物理机CPU数
    M = serverObj.server_info['MEM']# 物理机MEM数
    bw = M * 1.0 / C# 物理机权重
    
#######################################
    print 'pw=%.2f,bw=%.2f'%(pw,bw)
    #
    num = max(vm_cpu_size*1.0/C,vm_mem_size*1.0/M)
    print 'num=%d'%(ceil(num))
    
    print 'cpu%%=%.2f mem%%=%.2f'%(vm_cpu_size*100.0/(num*C),
                                   vm_mem_size*100.0/(num*M))
#######################################    
    
    # 创建最小量的虚拟机，原来集群中就存在一台，需要减一台
    serverObj.new_physic_machine(num=num - 1)
    
    
    
    # 获取CPU从大到小，权重都
    pick_func = vmPicker.get_vm_by_cpu
    dirt = cpu    
    start=len(dirt)-1
    end=-1
    step=-1
    order=0

    for i in range(start,end,step):
        tmp = pick_func(dirt[i],order)
        if tmp != None:
            vm_orders[0].extend(tmp[0])
            vm_orders[1].extend(tmp[1])

    if opt_target=='CPU':opt_index=0
    elif opt_target=='MEM':opt_index=1
    else: opt_index=2

    vm_type_size = len(vm_orders[0])
    if vm_type_size ==0:return # 无装配项，结束
    for vm_index in range(vm_type_size):
        vm_type = vm_orders[0][vm_index]
        vm_cot = vm_orders[1][vm_index]
        pm_size = serverObj.pm_size
        for rept in range(vm_cot):
            in_id  = -1
            max_opt=-1
            for pm_id in range(pm_size):
                ok,re_items = serverObj.test_put_vm(pm_id, vm_type)
                if not ok:continue
                if  max_opt<re_items[opt_index]:
                    max_opt = re_items[opt_index]
                    in_id = pm_id
                            
            if in_id<0 : # 在现有的物理机中无法安排该虚拟机
                pm_size = serverObj.new_physic_machine()
                re_items = serverObj.put_vm(pm_size - 1, vm_type)
                if re_items == None:
                    raise ValueError('ENDLESS LOOP ! ')
            else:
                serverObj.put_vm(in_id, vm_type)

    return (vm_cpu_size * 100.0 / (num * C),vm_mem_size * 100.0 / (num * M))



#################################################优化方案########################################

def search_maximum_way1(caseInfo, predict_result):
    global res_use_pro
    global vm_size
    global vm
    global pm_size
    global pm
    global try_result
    global other_res_use_pro
    global vm_map
    vm_size, vm, pm_size, pm, res_use_pro, other_res_use_pro, _ = packing_utils.pack_api(caseInfo, predict_result)
    pading_que = []

    # 搜索优先级
    if caseInfo.opt_target == 'CPU':
        pading_que = [1.0, 2.0, 4.0]
    else:
        pading_que = [4.0, 2.0, 1.0]

    # 根据数量初始化队列
    # vm_que=init_que(caseInfo)

    try_result = copy.deepcopy(predict_result)

    end_vm_pos = 0
    # 找到第一个非0位[1,15]
    for vm_type_index in range(len(VM_TYPE_DIRT) - 1, -1, -1):
        if try_result.has_key(VM_TYPE_DIRT[vm_type_index]) and try_result[VM_TYPE_DIRT[vm_type_index]] > 0:  # 键值对存在
            end_vm_pos = vm_type_index
            break
    for que in range(3):
        # 在有数量的区间内填充[1,8]
        for vm_type in range(end_vm_pos, -1, -1):
            if try_result.has_key(VM_TYPE_DIRT[vm_type]) and VM_PARAM[VM_TYPE_DIRT[vm_type]][2] == pading_que[
                que]:  # 键值对存在,C/M比相等
                if try_result[VM_TYPE_DIRT[vm_type]] > 0:
                    result_modify1(try_result, caseInfo, 1, VM_TYPE_DIRT[vm_type], vm_map)
                    result_modify1(try_result, caseInfo, -1, VM_TYPE_DIRT[vm_type], vm_map)
                else:
                    # 找到非0的,最大,虚拟机
                    result_modify1(try_result, caseInfo, 1, VM_TYPE_DIRT[vm_type], vm_map)


def search_maximum_way2(caseInfo, predict_result):
    global res_use_pro
    global vm_size
    global vm
    global pm_size
    global pm
    global try_result
    global other_res_use_pro
    vm_size, vm, pm_size, pm, res_use_pro, other_res_use_pro = packing_utils.pack_api(caseInfo, predict_result)
    pading_que = []

    # 搜索优先级
    if caseInfo.opt_target == 'CPU':
        pading_que = [1.0, 2.0, 4.0]
    else:
        pading_que = [4.0, 2.0, 1.0]

    # 根据数量初始化队列
    # vm_que=init_que(caseInfo)

    # 震荡范围
    value_range = 3
    # 范围表
    data_range = [[value_range] * caseInfo.vm_types_size]
    # 虚拟机类型
    vm_type = caseInfo.vm_types
    # 虚拟机震荡表
    vm_range = dict(zip(vm_type, data_range))

    try_result = copy.deepcopy(predict_result)
    end_vm_pos = 0
    # 找到第一个非0位[1,15]
    for vm_type_index in range(len(VM_TYPE_DIRT) - 1, -1, -1):
        if try_result.has_key(VM_TYPE_DIRT[vm_type_index]) and try_result[VM_TYPE_DIRT[vm_type_index]] > 0:  # 键值对存在
            end_vm_pos = vm_type_index
            break
    for que in range(3):
        # 在有数量的区间内填充[1,8]
        for vm_type in range(end_vm_pos, -1, -1):
            if try_result.has_key(VM_TYPE_DIRT[vm_type]) and VM_PARAM[VM_TYPE_DIRT[vm_type]][2] == pading_que[
                que]:  # 键值对存在,C/M比相等
                # 数量
                if try_result[VM_TYPE_DIRT[vm_type]] > 0:
                    result_modify1(try_result, caseInfo, 1, VM_TYPE_DIRT[vm_type])
                    result_modify1(try_result, caseInfo, -1, VM_TYPE_DIRT[vm_type])
                else:
                    # 找到非0的,最大,虚拟机
                    result_modify1(try_result, caseInfo, 1, VM_TYPE_DIRT[vm_type])


def result_modify1(predict_result, caseInfo, try_value, vm_type, try_vm_map):
    '''
    :param predict_result: 虚拟机预测结果 贪心搜索局部优解
    :param caseInfo: 训练集信息
    :param try_value: 尝试值
    :param vm_type: 虚拟机类型
    :return:
    '''
    global other_res_use_pro
    global res_use_pro
    global vm_size
    global vm
    global pm_size
    global pm
    global try_result
    global vm_map
    try_predict = copy.deepcopy(predict_result)
    try_vm_map = copy.deepcopy(vm_map)
    try_predict[vm_type][0] = try_predict[vm_type][0] + try_value
    if try_predict[vm_type][0] < 0:  # 小于0没有意义
        return
    try_vm_size, try_vm, try_pm_size, try_pm, try_res_use_pro, try_other_res_use_pro, _ = packing_utils.pack_api(
        caseInfo, try_predict)
    if try_res_use_pro > res_use_pro and try_pm_size <= pm_size:  # 如果结果优,物理机数量相等或者 【更小,利用率更高 】保存最优结果
        vm_size, vm, pm_size, pm, res_use_pro, other_res_use_pro = try_vm_size, try_vm, try_pm_size, try_pm, try_res_use_pro, try_other_res_use_pro
        try_result = try_predict
        try_vm_map[vm_type] += try_value
        vm_map = try_vm_map
        # 继续深度搜索
        result_modify1(try_predict, caseInfo, try_value, vm_type, try_vm_map)
    elif try_res_use_pro == res_use_pro and try_other_res_use_pro > other_res_use_pro:  # 如果没有当前的好,则返回
        vm_size, vm, pm_size, pm, res_use_pro, other_res_use_pro = try_vm_size, try_vm, try_pm_size, try_pm, try_res_use_pro, try_other_res_use_pro
        try_result = try_predict
        try_vm_map[vm_type] += try_value
        vm_map = try_vm_map
        # 继续深度搜索
        result_modify1(try_predict, caseInfo, try_value, vm_type, try_vm_map)
    else:
        return


def result_smooth(vm_size, vm, pm_size, pm, caseInfo, pm_free):
    '''
    平滑填充结果集
    :param vm:虚拟机列表
    :param pm_size:虚拟机数量
    :param pm:物理机列表
    :param caseInfo:数据对象
    :return:
    '''
    vm_types = caseInfo.vm_types
    res_use_pro = 0.0
    other_res_use_pro = 0.0
    VM_QUE = []
    free_cpu = 0.0
    free_mem = 0.0
    # 初始化填充队列
    if caseInfo.opt_target == 'CPU':
        VM_QUE = VM_CPU_QU
        res_use_pro = caseInfo.CPU * pm
        other_res_use_pro = caseInfo.MEM * pm
    else:
        VM_QUE = VM_MEM_QU
        res_use_pro = caseInfo.MEM * pm
        other_res_use_pro = caseInfo.CPU * pm

    epoch = 2
    # 遍历物理机
    for i in range(pm_size):
        M_C = 0.0
        # 进行多轮赋值,防止漏空
        for e in range(epoch):  # CPU 内存均有空间
            if pm_free[i][0] and pm_free[i][1]:
                # 计算占比
                M_C = computer_MC(pm_free[i])
                while (M_C >= 1 and pm_free[i][0] and pm_free[i][1]):  # CPU 内存均有空间
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
                                # 列表中数量添加
                                vm[VM_TYPE_DIRT[vm_type_index]] += 1
                                # 物理机列表中添加
                                if isContainKey(pm[i], VM_TYPE_DIRT[vm_type_index]):
                                    pm[i][VM_TYPE_DIRT[vm_type_index]] += 1
                                else:
                                    pm[i][VM_TYPE_DIRT[vm_type_index]] = 1
                                # 剪切空闲空间数
                                pm_free[i][0] = pm_free[i][0] - VM_PARAM[VM_TYPE_DIRT[vm_type_index]][0]
                                pm_free[i][1] = pm_free[i][1] - VM_PARAM[VM_TYPE_DIRT[vm_type_index]][1]
                                # 无空闲资源,则跳出循环
                                if pm_free[i][0] == 0 or pm_free[i][1] == 0:
                                    break
                    # 占比减半
                    M_C = M_C / 2.0
        free_cpu += pm_free[i][0]
        free_mem += pm_free[i][1]
        print('i:cpu:%d mem:%d' % (pm_free[i][0], pm_free[i][1]))
    if caseInfo.opt_target == 'CPU':
        res_use_pro = free_cpu / (caseInfo.CPU * pm_size)
        other_res_use_pro = free_mem / (caseInfo.MEM * pm_size)
    else:
        res_use_pro = free_mem / (caseInfo.MEM * pm_size)
        other_res_use_pro = free_cpu / (caseInfo.CPU * pm_size)

    res_use_pro = (1.0 - res_use_pro) * 100
    other_res_use_pro = (1.0 - other_res_use_pro) * 100
    return vm_size, vm, pm_size, pm, res_use_pro, other_res_use_pro


def res_average(vm_size, vm, pm_size, pm, res_use_pro, other_res_use_pro, pm_free, vm_map, caseInfo,predict_result):
    avg_predict_result=copy.deepcopy(predict_result)

    vm_types=caseInfo.vm_types

    avg_value=-1
    M_C=0.0
    if caseInfo.opt_target=='CPU':
        M_C=4.0
    else:
        M_C=1.0

    if res_use_pro<other_res_use_pro:
        for vm_type in vm_types:
            if VM_PARAM[vm_type][2]==M_C and avg_predict_result[vm_type][0]>=-avg_value:
                avg_predict_result[vm_type][0]+=avg_value

    return avg_predict_result


# 检查dict中是否存在key
def isContainKey(dic, key):
    return key in dic


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





#########################################
# 选择装配方案
used_func= pack_model
#########################################



