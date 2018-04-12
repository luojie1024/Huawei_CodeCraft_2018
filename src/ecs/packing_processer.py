# -*- coding: utf-8 -*-

'''
    装配模型，输入为预测模型输出的预测对象，
    在转配模型中可维护一个历史物理机集群状态对象，
    一个预测输入结果获取的picker对象，
    最终分配结果在Group对象中
'''

import ParamInfo
import math
import packing_model

# 选择在packing_model 中的装配方案
pack_function = packing_model.used_func


def pack_all(caseInfo, predict_result):
    '''
    装配模块对外接口，
    caseInfo 为案例对象
    predict_result 为预测模块结果
    返回vm_size,vm,pm_size,pm 用于生成结果文件
    '''
    group = MachineGroup(caseInfo)
    picker = VmPicker(predict_result)
    pack_function(picker, group, caseInfo.opt_target)
    vm_size, vm = picker.to_origin_desc()
    pm_size, pm = group.to_description()
    pm_free=group.get_pm_free()
    res_use_pro = group.get_res_used_pro(caseInfo.opt_target)
    other_res_use_pro = group.get_other_res_used_pro(caseInfo.opt_target)

    print(group.to_usage_status())
    return vm_size, vm, pm_size, pm, res_use_pro, other_res_use_pro,pm_free


class MachineGroup():
    '''
    静态集群状态类
    '''

    # 计数
    empty = 0

    # 集群中物理机参数
    machine_info = {'CPU': 0,  # u数
                    'MEM': 0,  # m数
                    'HDD': 0}  # h数

    # 物理机计数
    pm_size = 0
    # 各个物理机状态，存储值为
    # re_cpu:剩余u数，re_mem:剩余m数，vm_size:当前物理机中虚拟机数
    # [pm_id->{re_cpu:cot,re_mem:cot,vm_size:cot},
    #  pm_id2->{re_cpu:cot,re_mem:cot,vm_size:cot...]
    PM_status = []

    # 当前集群中虚拟机计数
    vm_size = 0
    # 虚拟机存储状态，对应的存储为
    # {vm_type:cot,vm_type2:cot...}
    VM = {}

    # 物理机存储状态，对应存储值
    # [pm_id->{vm_type:cot,vm_type2:cot....},
    #  pm_id2->{vm_type:cot,vm_type2:cot...}...]
    PM = []

    #剩余资源表
    PM_Free=[]

    def __init__(self, caseInfo):
        '''
        初始化集群，创建一个物理机，并初始化相关参数
        '''
        self.vm_size = 0
        self.PM = []
        self.VM = {}
        self.PM_status = []
        self.pm_size = 0
        self.empty = 0
        self.PM_Free=[]
        self.machine_info = {'CPU': 0,  # u数
                             'MEM': 0,  # m数
                             'HDD': 0}  # h数

        self.machine_info['CPU'] = caseInfo.CPU
        self.machine_info['MEM'] = caseInfo.MEM
        self.machine_info['HDD'] = caseInfo.HDD
        self.new_physic_machine()
        pass

    def new_physic_machine(self, num=1):
        '''
        创建num个新的物理机，返回物理机数量
        '''
        while num > 0:
            self.pm_size += 1
            npm = {
                're_cpu': self.machine_info['CPU'],
                're_mem': self.machine_info['MEM'],
                'vm_size': 0
            }
            self.PM_status.append(npm)
            self.PM.append({})
            num -= 1
        return self.pm_size

    def test_put_vm(self, pm_id, vm_type):
        '''
        测试能否放置虚拟机
        '''
        if pm_id is None or \
                pm_id < 0 or pm_id >= self.pm_size:
            raise ValueError('error pm_id=', pm_id)
        vm_cpu, vm_mem = ParamInfo.VM_PARAM[vm_type][:2]
        pmstatus = self.PM_status[pm_id]
        re_cpu = pmstatus['re_cpu'] - vm_cpu
        re_mem = pmstatus['re_mem'] - vm_mem
        if re_cpu >= 0 and re_mem >= 0:
            return (True, [re_cpu, re_mem])
        else:
            return (False, [re_cpu, re_mem])

    def put_vm(self, pm_id, vm_type):
        '''
        放置一个虚拟机，
        pm_id为物理机ID，vm_type为虚拟机类型名
        如果放置成功 返回放置后的物理机(re_cpu,re_mem)，
        因空间不足放置失败返回None
        '''
        if pm_id is None or \
                pm_id < 0 or pm_id >= self.pm_size:
            raise ValueError('error pm_id=', pm_id)
        vm_cpu, vm_mem = ParamInfo.VM_PARAM[vm_type][:2]
        pmstatus = self.PM_status[pm_id]
        re_cpu = pmstatus['re_cpu'] - vm_cpu
        re_mem = pmstatus['re_mem'] - vm_mem
        if re_cpu >= 0 and re_mem >= 0:
            self.empty += 1
            pmstatus['re_cpu'] = re_cpu
            pmstatus['re_mem'] = re_mem
            pmstatus['vm_size'] += 1
            self.vm_size += 1
            if vm_type not in self.VM.keys():
                self.VM[vm_type] = 0
            self.VM[vm_type] += 1
            pm = self.PM[pm_id]
            if vm_type not in pm.keys():
                pm[vm_type] = 0
            pm[vm_type] += 1
            return (re_cpu, re_mem)
        return None  # 超分返回

    def to_description(self):
        '''
        统计当前PM一个描述结果
        返回当前pm_size PM
        '''
        if self.empty != 0:
            return self.pm_size, self.PM
        else:
            return 0, self.PM

    def get_res_used_pro(self, opt_target='CPU'):
        '''
        :param opt_target:资源优化目标
        :return: 返回总的资源优化利用率
        '''
        # 获取最大资源数
        res_max = self.machine_info[opt_target]
        # 已经使用的资源状态
        usage = self.PM_status

        res_used = 0
        for i in range(self.pm_size):
            # 单个物理机资源使用率
            if opt_target == 'CPU':
                res_used += res_max - usage[i]['re_cpu']
            elif opt_target == 'MEM':
                res_used += res_max - usage[i]['re_mem']

        # 返回物理机的资源使用率
        return res_used * 100 / (res_max * self.pm_size)

    def get_other_res_used_pro(self, opt_target='MEM'):
        '''
        :param opt_target:另外一个资源的利用率
        :return: 返回总的资源优化利用率
        '''
        if opt_target == 'CPU':
            opt_target = 'MEM'
        else:
            opt_target = 'CPU'
        # 获取最大资源数
        res_max = self.machine_info[opt_target]
        # 已经使用的资源状态
        usage = self.PM_status

        res_used = 0
        for i in range(self.pm_size):
            # 单个物理机资源使用率
            if opt_target == 'CPU':
                res_used += res_max - usage[i]['re_cpu']
            elif opt_target == 'MEM':
                res_used += res_max - usage[i]['re_mem']

        # 返回物理机的资源使用率
        return res_used * 100 / (res_max * self.pm_size)

    def get_last_res_used_pro(self, opt_target='CPU'):
        '''
        :param opt_target:资源优化目标
        :return: 返回最后一个物理机的资源利用率
        '''
        # 获取最大资源数
        res_max = self.machine_info[opt_target]
        # 已经使用的资源状态
        usage = self.PM_status
        # 获取资源使用率
        if opt_target == 'CPU':
            res_used = res_max - usage[self.pm_size - 1]['re_cpu']

        elif opt_target == 'MEM':
            res_used = res_max - usage[self.pm_size - 1]['re_mem']

        # 返回最后一台物理机的资源使用率
        return res_used * 100.0 / res_max

    def to_usage_status(self):
        '''
        生成当前集群中各个物理机的使用状态
        '''
        cpu_max = self.machine_info['CPU']
        mem_max = self.machine_info['MEM']
        usage = self.PM_status
        result = 'CPU:%d MEM:%d\n' % (cpu_max, mem_max)
        for i in range(self.pm_size):
            cpu_used = cpu_max - usage[i]['re_cpu']
            mem_used = mem_max - usage[i]['re_mem']
            vm_cot = usage[i]['vm_size']
            string = 'pm_id:%d cpu_used:%d(%.2f%%) ' % (i, cpu_used, cpu_used * 100.0 / cpu_max)
            string += 'mem_used:%d(%.2f%%) vm_cot:%d\n' % (mem_used, mem_used * 100.0 / mem_max, vm_cot)
            #保存剩余空间情况表
            self.PM_Free.append([cpu_max-cpu_used,mem_max-mem_used])
            result += string
        return result

    def get_pm_free(self):
        return self.PM_Free

################## end class MachineGroup ####################


class VmPicker():
    '''
    输入预测模型的预测结果，
    并维护一个权重与核心数级别的二维映射表，
    调用任何get_xxx 方法会时Picker中的虚拟机数减少，
    直到全部虚拟机被取完。
    二维表中原值为-1,未被预测虚拟机，
    大于等于0表示已被预测的虚拟机数量
    '''

    # 预测输入的原始数据
    origin_data = None
    # 原始输入描述
    origin_desc_table = {}
    origin_vm_size = 0

    # 虚拟机总数，非零虚拟机总数
    vm_size = 0

    # 虚拟机中cpu总数
    vm_cpu_size = 0

    # 虚拟机中mem总数
    vm_mem_size = 0

    # 预测虚拟机的在M/U权重与核心数级别
    # 上展开 shape=[3,5]
    #   CPU=1,2,4,8,16
    VM = [[-1, -1, -1, -1, -1],  # weight_1.0
          [-1, -1, -1, -1, -1],  # weight_2.0
          [-1, -1, -1, -1, -1]  # weight_4.0
          ]

    # 虚拟机类型名数组
    vm_types = ParamInfo.VM_TYPE_DIRT

    def __init__(self, predict_result):
        self.origin_data = predict_result
        self.init_picker(predict_result)
        self.vm_size, self.origin_desc_table = \
            self.to_description()
        self.origin_vm_size = self.vm_size
        pass

    def init_picker(self, predict_result):
        types = predict_result.keys()
        for vmtype in types:
            vsum = 0
            pre = predict_result[vmtype]
            vm_cpu, vm_mem, _ = ParamInfo.VM_PARAM[vmtype]
            for i in range(len(pre)):
                vsum += pre[i]
            self.vm_cpu_size += vm_cpu * vsum
            self.vm_mem_size += vm_mem * vsum
            windex, cindex = self.type2index(vmtype)
            self.VM[windex][cindex] = vsum
        pass

    def type2index(self, vm_type):
        tindex = self.vm_types.index(vm_type)
        windex = tindex % 3
        cindex = int(tindex / 3)
        return windex, cindex

    def index2type(self, windex, cindex):
        if windex < 0 or cindex < 0:
            raise ValueError('Error ', (windex, cindex))
        return self.vm_types[cindex * 3 + windex]

    def get_vm_by_index(self, windex, cindex):
        '''
        windex M/U权重的下标 cindex CPU数下标，
        若原先并没有预测则返回None,拿取失败
        若原先有预测但当前数量为0,返回-1,拿取失败，
        正常情况 返回 该虚拟机类型剩余量
        '''

        re_vm = self.VM[windex][cindex]
        if self.vm_size == -1 or re_vm == -1:
            return None
        elif self.vm_size == 0 or re_vm == 0:
            return -1
        else:
            re_vm -= 1
            self.vm_size -= 1
        self.VM[windex][cindex] = re_vm
        return re_vm
        pass

    def get_vm_by_wc(self, weight, cpu):
        '''
        通过虚拟机M/U权重和CPU数获取，
        若原先并没有预测则返回None,拿取失败
        若原先有预测但当前数量为0,返回-1,拿取失败，
        正常情况 返回 该虚拟机类型剩余量
        '''
        windex = int(math.log(weight, 2))
        cindex = int(math.log(cpu, 2))
        return self.get_vm_by_index(windex, cindex)
        pass

    def get_vm_by_type(self, vm_type):
        '''
        通过虚拟机类型名获取，
        若原先并没有预测则返回None,拿取失败
        若原先有预测但当前数量为0,返回-1,拿取失败，
        正常情况 返回 该虚拟机类型剩余量
        '''
        windex, cindex = self.type2index(vm_type)
        return self.get_vm_by_index(windex, cindex)

    def get_vm_by_mu_weight(self, mu_weight, order=0):
        '''
        获取某一M/U权重下所有虚拟机，并按照cpu数排序
        注意：调用该函数后，M/u权重下所有有预测结果的虚拟机计数都清为0
        mu_weight:权重值，[1，2，4]
        order:排序方法，order=1时按照cpu降序给出结果,
        其他按照cpu升序给出结果
        返回格式：存在值时[[vm_type1,vm_type2...],[cot1,cot2...]]
        无效时返回None
        '''
        result = [[],  # vm_type
                  []]  # cot
        windex = int(math.log(mu_weight, 2))
        start = 0
        end = 5
        step = 1

        if order == 1:
            start = 4
            end = -1
            step = -1
        for cindex in range(start, end, step):
            tmp = self.VM[windex][cindex]
            if tmp > 0:
                result[0].append(self.index2type(windex, cindex))
                result[1].append(tmp)
                self.VM[windex][cindex] = 0
                self.vm_size -= tmp
        if len(result[0]) == 0:
            return None
        return result

    def get_vm_by_cpu(self, cpu, order=0):
        '''
        获取某一CPU数值下所有虚拟机，并按照M/U权重数排序
        注意：调用该函数后，CPU数值下所有有预测结果的虚拟机计数都清为0
        CPU:核数，[1，2，4，8，16]
        order:排序方法，order=1时按照权重降序给出结果,
        其他按照权重升序给出结果
        返回格式：存在值时[[vm_type1,vm_type2...],[cot1,cot2...]]
        无效时返回None
        '''
        result = [[],  # vm_type
                  []]  # cot
        cindex = int(math.log(cpu, 2))
        start = 0
        end = 3
        step = 1

        if order == 1:
            start = 2
            end = -1
            step = -1
        for windex in range(start, end, step):
            tmp = self.VM[windex][cindex]
            if tmp > 0:
                result[0].append(self.index2type(windex, cindex))
                result[1].append(tmp)
                self.VM[windex][cindex] = 0
                self.vm_size -= tmp
        if len(result[0]) == 0:
            return None
        return result

    def origin_cpu_mem_sum(self):
        return self.vm_cpu_size, self.vm_mem_size

    def to_origin_desc(self):
        return self.origin_vm_size, self.origin_desc_table
        pass

    def to_description(self):
        '''
        统计当前VM一个描述结果
        返回当前vm_size vm_desc_table
        '''
        new_desc_table = {}
        vmsum = 0
        flag = True
        for i in range(3):
            for j in range(5):
                tmp = self.VM[i][j]
                if tmp != -1:
                    flag = False
                    vmsum += tmp
                    new_desc_table[self.index2type(i, j)] = tmp
        if flag:
            vmsum = -1
        return vmsum, new_desc_table

    pass
