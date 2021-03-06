# -*- coding: utf-8 -*-


import const_map
import math
import packing_method

# 选择在packing_model 中的装配方案
pack_function = packing_method.used_func


def pack_api(dataObj, predict_result):
    '''
    装配接口
    :param dataObj: 数据对象
    :param predict_result: 预测结果
    '''
    group = ServerObj(dataObj)
    picker = VmWorker(predict_result)
    pack_function(picker, group, dataObj.opt_target)
    vm_size, vm = picker.to_origin_desc()
    pm_size, pm = group.to_description()
    pm_free = group.get_pm_free()
    res_use_pro = group.get_res_used_pro(dataObj.opt_target)
    other_res_use_pro = group.get_other_res_used_pro(dataObj.opt_target)

    print(group.to_usage_status())
    return vm_size, vm, pm_size, pm, res_use_pro, other_res_use_pro, pm_free


class ServerObj():
    # 计数
    empty = 0

    # 集群中物理机参数
    server_info = {'CPU': 0,  # u数
                   'MEM': 0,  # m数
                   'HDD': 0}  # h数

    # 物理机计数量
    pm_size = 0

    PM_status = []

    # 当前集群中虚拟机计数
    vm_size = 0
    # 虚拟机存储状态，对应的存储为
    VM = {}

    # 物理机存储状态，对应存储值
    PM = []

    # 剩余资源表
    PM_Free = []

    def __init__(self, dataObj):
        '''
        初始化
        '''
        self.vm_size = 0
        self.PM = []
        self.VM = {}
        self.PM_status = []
        self.pm_size = 0
        self.empty = 0
        self.PM_Free = []
        self.server_info = {'CPU': 0,
                            'MEM': 0,
                            'HDD': 0}

        self.server_info['CPU'] = dataObj.CPU
        self.server_info['MEM'] = dataObj.MEM
        self.server_info['HDD'] = dataObj.HDD
        self.new_physic_machine()
        pass

    def new_physic_machine(self, num=1):
        '''
        创建num个新的物理机，返回物理机数量
        '''
        while num > 0:
            self.pm_size += 1
            npm = {
                're_cpu': self.server_info['CPU'],
                're_mem': self.server_info['MEM'],
                'vm_size': 0
            }
            self.PM_status.append(npm)
            self.PM.append({})
            num -= 1
        return self.pm_size

    def test_put_vm(self, pm_id, vm_type):
        '''
        测试能否放置虚拟机
        :param pm_id: 物理机id
        :param vm_type: 虚拟机类型
        :return: 剩余资源数
        '''
        if pm_id is None or \
                pm_id < 0 or pm_id >= self.pm_size:
            raise ValueError('error pm_id=', pm_id)
        vm_cpu, vm_mem = const_map.VM_PARAM[vm_type][:2]
        pmstatus = self.PM_status[pm_id]
        re_cpu = pmstatus['re_cpu'] - vm_cpu
        re_mem = pmstatus['re_mem'] - vm_mem
        if re_cpu >= 0 and re_mem >= 0:
            return (True, [re_cpu, re_mem])
        else:
            return (False, [re_cpu, re_mem])

    def put_vm(self, pm_id, vm_type):
        '''
        :param pm_id:物理机id
        :param vm_type: 虚拟机类型
        :return:
        '''
        if pm_id is None or \
                pm_id < 0 or pm_id >= self.pm_size:
            raise ValueError('error pm_id=', pm_id)
        vm_cpu, vm_mem = const_map.VM_PARAM[vm_type][:2]
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
        res_max = self.server_info[opt_target]
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
        res_max = self.server_info[opt_target]
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
        res_max = self.server_info[opt_target]
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
        cpu_max = self.server_info['CPU']
        mem_max = self.server_info['MEM']
        usage = self.PM_status
        result = 'CPU:%d MEM:%d\n' % (cpu_max, mem_max)
        for i in range(self.pm_size):
            cpu_used = cpu_max - usage[i]['re_cpu']
            mem_used = mem_max - usage[i]['re_mem']
            vm_cot = usage[i]['vm_size']
            string = 'pm_id:%d cpu_used:%d(%.2f%%) ' % (i, cpu_used, cpu_used * 100.0 / cpu_max)
            string += 'mem_used:%d(%.2f%%) vm_cot:%d\n' % (mem_used, mem_used * 100.0 / mem_max, vm_cot)
            # 保存剩余空间情况表
            self.PM_Free.append([cpu_max - cpu_used, mem_max - mem_used])
            result += string
        return result

    def get_pm_free(self):
        return self.PM_Free


################## end class Server ####################


class VmWorker():
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
    # 上展开 shape=[3,6]
    #   CPU=1,2,4,8,16,32
    VM = [[-1, -1, -1, -1, -1, -1],  # weight_1.0
          [-1, -1, -1, -1, -1, -1],  # weight_2.0
          [-1, -1, -1, -1, -1, -1]  # weight_4.0
          ]

    # 虚拟机类型名数组
    vm_types = const_map.VM_TYPE_DIRT

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
            vm_cpu, vm_mem, _ = const_map.VM_PARAM[vmtype]
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
        :param windex:
        :param cindex:
        :return:
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
        :param weight:
        :param cpu:
        :return:
        '''
        windex = int(math.log(weight, 2))
        cindex = int(math.log(cpu, 2))
        return self.get_vm_by_index(windex, cindex)
        pass

    def get_vm_by_type(self, vm_type):
        windex, cindex = self.type2index(vm_type)
        return self.get_vm_by_index(windex, cindex)

    def get_vm_by_mu_weight(self, mu_weight, order=0):
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

    def to_description(self):
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
