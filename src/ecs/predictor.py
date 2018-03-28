# -*- coding: utf-8 -*-

import CaseProcess
import packing_processer
import prediction_processer


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
    caseInfo = CaseProcess.CaseInfo(input_lines,ecs_lines)

    #使用RNN进行预测
    # predict_result = train_RNN(caseInfo)

    # 预测数据 Step 03
    predict_result = prediction_processer.predict_all(caseInfo)

    # predict_result = prediction_processer.predict_one(1,caseInfo)
#     predict_result={'flavor1':[1,2,3,4,5,6,7],
#                     'flavor2':[0,0,0,0,0,0,0],
#                     'flavor3':[10,2,10,4,10,6,10],
#                     'flavor4':[1,0,0,1,0,0,1],
#                     'flavor5':[1,0,3,0,5,0,7]}
    vm_size,vm,pm_size,pm = packing_processer.pack_all(caseInfo, predict_result)
    
    result = result_to_list(vm_size, vm, pm_size, pm)
    print(result)
    return result


def result_to_list(vm_size,vm,pm_size,pm):
    '''
    由预测和分配生成结果
    vm：{vm_type:cot...}
    pm[{vm_type:cot,vm_type2:cot2...}...]
    '''
    end_str=''
    result=[]
    result.append(str(vm_size)+end_str)
    for index in vm.keys():
        item = vm[index]
        tmp = index +' '+str(item)+end_str
        result.append(tmp)
        
    result.append(end_str)
    #TODO
    result.append(str(pm_size)+end_str)
    for pm_id in range(len(pm)):
        tmp = str(pm_id+1)
        pmone = pm[pm_id]
        if len(pmone.keys())==0:
            continue
        for index in pmone.keys():
            item = pmone[index]
            tmp += ' '+index+' '+str(item)
        tmp+=end_str
        result.append(tmp)
    return result
    




