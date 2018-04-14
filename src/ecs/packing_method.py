# -*- coding: utf-8 -*-
from math import ceil

def pack_model1(vmPicker,machineGroup,opt_target='CPU'):
    '''
    具体装配方案1,packing1,
    先预先开某一数目的物理机，然后查找在物理机中能放得进并且优化容量最大的机器放入，
    若无法放入则重新开一个物理机。
    装配顺序：
    （c,tc）:优先装载CPU多的并且M/U权重小的虚拟机,
    （c,tm）:优先装载CPU多的并且M/U权重小的虚拟机,
    （m,tc）:优先装载CPU多的并且M/U权重小的虚拟机,
    （m,tm）:优先装载CPU多的并且M/U权重小的虚拟机,
    优先装载CPU多的并且M/U权重小的虚拟机
    vmPicker:
    machineGroup:
    opt_target:优化目标[CPU,MEM],默认CPU优化
    '''
    # 获得放置顺序
    vm_orders = [[], # vm_types
                 []] # cot
    weightes = [1,2,4]
    cpu = [1,2,4,8,16]

    vm_cpu_size,vm_mem_size = vmPicker.origin_cpu_mem_sum()
    
    if vm_cpu_size == 0: return  # 无需装装配， 结束
    
    pw =vm_mem_size*1.0 / vm_cpu_size
    
    C = machineGroup.machine_info['CPU']# 物理机CPU数
    M = machineGroup.machine_info['MEM']# 物理机MEM数
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
    machineGroup.new_physic_machine(num=num-1)
    
    
    
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
        pm_size = machineGroup.pm_size
        for rept in range(vm_cot):
            in_id  = -1
            max_opt=-1
            for pm_id in range(pm_size):
                ok,re_items = machineGroup.test_put_vm(pm_id, vm_type)
                if not ok:continue
                if  max_opt<re_items[opt_index]:
                    max_opt = re_items[opt_index]
                    in_id = pm_id
                            
            if in_id<0 : # 在现有的物理机中无法安排该虚拟机
                pm_size = machineGroup.new_physic_machine()
                re_items = machineGroup.put_vm(pm_size-1,vm_type)
                if re_items == None:
                    raise ValueError('ENDLESS LOOP ! ')
            else:
                machineGroup.put_vm(in_id,vm_type)

    # return (vm_cpu_size * 100.0 / (num * C),vm_mem_size * 100.0 / (num * M))
############################## end model5 ###############################












#########################################
# 选择装配方案
used_func= pack_model1
#########################################



