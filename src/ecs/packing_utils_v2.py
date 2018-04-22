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
    picker = VmWorker(predict_result)
    group = ServerObj(dataObj, picker.origin_cpu_mem_sum())
    pack_function(picker, group)
    vm_size, vm = picker.to_origin_desc()
    pm_size, pm, pm_name = group.to_description()
    res_use = group.get_res_used_pro()
    pm_free = group.get_pm_free()
    print(group.to_usage_status())
    return vm_size, vm, pm_size, pm, pm_name, res_use, pm_free


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
    PM_name = []

    # 剩余资源表
    PM_Free = []

    # 当前指向的物理机id 索引从0开始
    PM_ID = 0

    # 目标比例
    direction = []

    # 剩余的cpu数量
    lave_cpu_sum = 0
    # 剩余的mem数量
    lave_mem_sum = 0

    # vm总共需要cpu数量
    need_cpu_sum = 0
    # vm总共需要mem数量
    need_mem_sum = 0

    # 物理机cpu总数
    pm_cpu_sum = 0
    # 物理机内存总算
    pm_mem_sum = 0

    def __init__(self, dataObj, vm_res):
        '''
        初始化
        '''
        self.vm_size = 0
        self.PM = []
        self.VM = {}
        self.PM_status = []
        self.pm_size = 0
        self.PM_ID = -1
        self.empty = 0
        self.PM_Free = []
        self.server_info = {}
        self.server_info = dataObj.pm_type_list
        self.direction = [0.25, 0.5, 1]
        self.lave_mem_sum = vm_res[1]
        self.lave_cpu_sum = vm_res[0]
        self.need_mem_sum = vm_res[1]
        self.need_cpu_sum = vm_res[0]
        self.pm_cpu_sum = 0
        self.pm_mem_sum = 0
        self.PM_name = []

    def new_physic_machine(self, pm_type):
        '''
        创建物理机
        :param pm_type:虚拟机类型
        :return:
        '''
        C_M = const_map.PM_TYPE[pm_type]['CPU'] / float(const_map.PM_TYPE[pm_type]['MEM'])
        re_cpu = const_map.PM_TYPE[pm_type]['CPU']
        re_mem = const_map.PM_TYPE[pm_type]['MEM']
        temp = {
            'pm_type': pm_type,
            'C_M': C_M,
            're_cpu': re_cpu,
            're_mem': re_mem,
            'vm_size': 0
        }
        self.PM_status.append(temp)
        self.PM.append({})
        self.pm_size += 1
        self.PM_ID += 1

        # 保存现在总的物理资源开辟数量
        self.pm_cpu_sum += re_cpu
        self.pm_mem_sum += re_mem
        # 存储物理机名字
        self.PM_name.append(pm_type)
        print 'apply pm：%s , C/M=%.2f\n' % (pm_type, C_M)
        return self.PM_ID

    def get_nearest_distance(self, c_m):
        '''
        获取最接近c_m的优化目标
        :param c_m:
        :return:
        '''
        min_distance_target = 1
        distance = 1
        for i in range(len(self.direction)):
            # 距离更接近
            if abs(c_m - self.direction[i]) < distance:
                distance = abs(c_m - self.direction[i])
                min_distance_target = self.direction[i]
        return min_distance_target

    def get_pm_c_m(self, pm_id):
        '''
        返回指定物理机的c/m
        :param pm_id:
        :return:
        '''
        c_m = self.PM_status[pm_id]['C_M']
        return c_m

    def get_lave_cpu_mem_sum(self):
        '''
        获取当前cpu mem的数量
        :return:
        '''
        return self.lave_cpu_sum, self.lave_mem_sum

    def get_sum_C_M(self):
        return self.lave_cpu_sum * 1.0 / self.lave_mem_sum

    def is_free(self, pm_id):
        '''
        判断是否还没放满
        :param pm: 物理机编号
        :return: 状态
        '''
        re_cpu = self.PM_status[pm_id]['re_cpu']
        re_mem = self.PM_status[pm_id]['re_mem']
        if re_cpu > 0 and re_mem > 0:
            return True
        else:
            return False

    def get_pm_cpu_mem(self, pm_id):
        '''
        返回指定物理机的cpu 内存剩余空间
        :param pm_id:
        :return:
        '''
        re_cpu = self.PM_status[pm_id]['re_cpu']
        re_mem = self.PM_status[pm_id]['re_mem']
        return re_cpu, re_mem

    def test_put_vm(self, pm_id, vm_type):
        '''
        测试能否放置虚拟机
        :param pm_id: 物理机id
        :param vm_type: 虚拟机类型
        :return: 剩余资源数
        '''
        # 数据异常
        if pm_id is None or \
                pm_id < 0 or pm_id >= self.pm_size:
            raise ValueError('error pm_id=', pm_id)
        vm_cpu, vm_mem = const_map.VM_PARAM[vm_type][:2]
        # 从物理机状态表中获取参数
        pmstatus = self.PM_status[pm_id]
        re_cpu = pmstatus['re_cpu'] - vm_cpu
        re_mem = pmstatus['re_mem'] - vm_mem
        if re_cpu == 0 or re_mem == 0:
            c_m = 0
        else:
            c_m = re_cpu * 1.0 / re_mem
        # 返回能否放置,并返回放置后的剩余空间大小
        if re_cpu >= 0 and re_mem >= 0:
            return (True, [re_cpu, re_mem, c_m])
        else:
            return (False, [re_cpu, re_mem, c_m])

    def put_vm(self, pm_id, vm_type):
        '''
        :param pm_id:物理机id
        :param vm_type: 虚拟机类型
        :return:
        '''
        if pm_id is None or \
                pm_id < 0 or pm_id >= self.pm_size:
            raise ValueError('error pm_id=', pm_id)
        # 获取资源数
        vm_cpu, vm_mem = const_map.VM_PARAM[vm_type][:2]
        # 获取参数状态
        pmstatus = self.PM_status[pm_id]
        re_cpu = pmstatus['re_cpu'] - vm_cpu
        re_mem = pmstatus['re_mem'] - vm_mem

        # 剩余总数计算
        self.lave_cpu_sum -= vm_cpu
        self.lave_mem_sum -= vm_mem

        # 资源充足,分配
        if re_cpu >= 0 and re_mem >= 0:
            self.empty += 1
            pmstatus['re_cpu'] = re_cpu
            pmstatus['re_mem'] = re_mem
            # 计算c/m比例
            if re_cpu == 0 or re_mem == 0:
                c_m = 0
            else:
                c_m = re_cpu * 1.0 / re_mem
            pmstatus['C_M'] = c_m
            pmstatus['vm_size'] += 1
            self.vm_size += 1
            # 记录虚拟机种类数量
            if vm_type not in self.VM.keys():
                self.VM[vm_type] = 0
            self.VM[vm_type] += 1
            pm = self.PM[pm_id]
            # 记录物理机种类数量
            if vm_type not in pm.keys():
                pm[vm_type] = 0
            pm[vm_type] += 1
            return (re_cpu, re_mem)
        return None  # 超分返回

    def to_description(self):
        if self.empty != 0:
            return self.pm_size, self.PM, self.PM_name
        else:
            return 0, self.PM, self.PM_name

    def get_res_used_pro(self):
        '''
        :return: 返回资源使用率
        '''
        cpu_use = self.need_cpu_sum * 1.0 / self.pm_cpu_sum
        mem_use = self.need_mem_sum * 1.0 / self.pm_mem_sum
        use = cpu_use * 0.5 + mem_use * 0.5
        # 返回物理机的资源使用率
        # return cpu_use, mem_use, use
        return use

    def to_usage_status(self):
        '''
        生成当前集群中各个物理机的使用状态
        '''
        result = ''
        usage = self.PM_status
        # result = 'CPU:%d MEM:%d\n' % (cpu_max, mem_max)
        for i in range(self.pm_size):
            pm_type = usage[i]['pm_type']
            cpu_max = self.server_info[pm_type]['CPU']
            mem_max = self.server_info[pm_type]['MEM']
            cpu_used = cpu_max - usage[i]['re_cpu']
            mem_used = mem_max - usage[i]['re_mem']
            cpu_usage_rate = cpu_used * 100.0 / cpu_max
            mem_usage_rate = mem_used * 100.0 / mem_max
            total_usage_rate = cpu_usage_rate * 0.5 + mem_usage_rate * 0.5
            vm_cot = usage[i]['vm_size']
            string = 'pm_id:%d \t cpu_used:%d(%.2f%%)\t' % (i, cpu_used, cpu_usage_rate)
            string += 'mem_used:%d(%.2f%%)\t' % (mem_used, mem_usage_rate)
            string += 'total_used:(%.2f%%)\tvm_cot:%d\n' % (total_usage_rate, vm_cot)
            # 保存剩余空间情况表
            self.PM_Free.append([cpu_max - cpu_used, mem_max - mem_used])
            result += string
        return result

    def get_pm_free(self):
        return self.PM_Free

    def is_packing(self):
        if self.lave_cpu_sum == 0 or self.lave_mem_sum == 0:
            return False
        else:
            return True


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
        # 保存原始数据
        self.origin_data = predict_result
        # 初始化分拣对象
        self.init_worker(predict_result)

        self.vm_size, self.origin_desc_table = self.set_data_info()

        self.origin_vm_size = self.vm_size
        # 初始化实时的cpu mem数量
        self.cpu_sum = self.vm_cpu_size
        self.mem_sum = self.vm_mem_size
        pass

    def init_worker(self, predict_result):
        '''
        初始化分拣对象
        :param predict_result:预测结果
        '''
        types = predict_result.keys()
        # 遍历计算总共需要cpu mem 的数量
        for vmtype in types:
            vm_sum = 0
            pre_temp = predict_result[vmtype]
            vm_cpu, vm_mem, _ = const_map.VM_PARAM[vmtype]
            for i in range(len(pre_temp)):
                vm_sum += pre_temp[i]
            self.vm_cpu_size += vm_cpu * vm_sum
            self.vm_mem_size += vm_mem * vm_sum
            # 添加到数量列表
            row, col = self.type2index(vmtype)
            self.VM[row][col] = vm_sum

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

    def get_vm_order(self, cpu):
        '''
        :param cpu:CPU
        :return: 返回该cpu类型下所有比例队列
        '''
        result = {}
        col = int(math.log(cpu, 2))
        start = col
        end = -1
        step = -1
        temp_1 = [[], []]
        temp_2 = [[], []]
        temp_4 = [[], []]
        for col in range(start, end, step):
            if self.VM[0][col] != -1:
                temp_1[0].append(self.index2type(0, col))
                temp_1[1].append(self.VM[0][col])
            if self.VM[1][col] != -1:
                temp_2[0].append(self.index2type(1, col))
                temp_2[1].append(self.VM[1][col])
            if self.VM[2][col] != -1:
                temp_4[0].append(self.index2type(2, col))
                temp_4[1].append(self.VM[2][col])
        # 如果都为空,则无需放置
        if len(temp_1[0]) == 0 and len(temp_2[0]) == 0 and len(temp_4[0]) == 0:
            return result
        else:
            result['1.0'] = temp_1
            result['2.0'] = temp_2
            result['4.0'] = temp_4
            return result

    def get_vm_by_cpu(self, cpu, order=0):
        '''
        获得队列顺序
        :param cpu:
        :param order:
        :return:
        '''
        result = [[],  # vm_type
                  []]  # cot
        # 计算CPU所在列
        col = int(math.log(cpu, 2))
        start = 0
        end = 3
        step = 1

        # 从下往上取 1：4->1：1
        if order == 1:
            start = 2
            end = -1
            step = -1

        # 从上往下取 1：1->1：4
        for row in range(start, end, step):
            tmp = self.VM[row][col]
            if tmp > 0:
                result[0].append(self.index2type(row, col))
                result[1].append(tmp)
                self.VM[row][col] = 0
                self.vm_size -= tmp
        # 没有vm 返回None
        if len(result[0]) == 0:
            return None
        return result

    def origin_cpu_mem_sum(self):
        return self.vm_cpu_size, self.vm_mem_size

    def to_origin_desc(self):
        return self.origin_vm_size, self.origin_desc_table

    def set_data_info(self):
        '''
        设置虚拟机数量表
        计算虚拟机总数
        '''
        info_table = {}
        vm_sum = 0
        flag = True
        for i in range(len(self.VM)):  # 行
            for j in range(len(self.VM[2])):  # 列
                tmp = self.VM[i][j]
                if tmp != -1:
                    flag = False
                    vm_sum += tmp
                    info_table[self.index2type(i, j)] = tmp
        if flag:
            vm_sum = -1
        return vm_sum, info_table
