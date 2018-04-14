# -*- coding: utf-8 -*-
from math import ceil

def pack_model1(vmPicker, serverObj, opt_target='CPU'):
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
############################## end model1 ###############################












#########################################
# 选择装配方案
used_func= pack_model1
#########################################



