# -*- coding: utf-8 -*-
import copy
from datetime import timedelta
from datetime import datetime
import math

import ParamInfo


class CaseInfo(object):
    '''
    训练数据对象
    '''
    CPU=0# cpu数
    MEM=0# 内存存储量 单位Gb
    HDD=0# 硬盘存储量 单位Gb
    
    opt_target=''# 优化目标，值为CPU和MEM
    
    vm_types_size=0# 虚拟机类型数
    vm_types=[]# 虚拟机类型{list}
    
    time_grain = -1 # 预测时间粒度
    date_range_size=0# 需要预测多少天数的数据
    data_range=[]#预测开始时间与结束时间，左闭右开
    
    # 训练数据中虚拟机、日期二维映射表 {虚拟机类型flavor1：{'日期':出现次数,.....]
    his_data={}

    # 训练特征表
    feature_list = []
    # 测试表
    predictor_list = []

    def __init__(self, origin_case_info,origin_train_data,predict_time_grain=ParamInfo.TIME_GRAIN_DAY):
        '''
        origin_data  predictor中的input_lines数组
        origin_train_data predictor中的ecs_lines数组
        初始化CaseInfo中的属性
        '''
        self.time_grain = predict_time_grain
        self.set_case_info(origin_case_info,predict_time_grain)
        self.set_his_data(origin_train_data, predict_time_grain)
        # 提取特征
        # self.set_feature_map()
        # self.set_predictor_map()
        pass
    
    def set_case_info(self,origin_case_info,predict_time_grain):
        '''
        更改 案例属性信息
        info[0]=CPU MEM HDD
        info[2]=vm_type_size
        info[3:(3+vm_type_size)]=vm_types
        info[4+vm_type_size]=opt_target
        info[6+vm_type_size]=start_time
        info[7+vm_type_size]=start_time
        '''
        if (origin_case_info is None) or \
            len(origin_case_info) < 2:
            raise ValueError('Error origin_case_info=',origin_case_info)
        
        # 处理 CPU MEM HDD
        tmp = origin_case_info[0].replace('\r\n','')
        tmps = tmp.split(' ')
        self.CPU=int(tmps[0])
        self.MEM=int(tmps[1])
        self.HDD=int(tmps[2])
        
        # 处理虚拟机类型
        tsize = int(origin_case_info[2].replace('\r\n',''))
        self.vm_types_size = tsize
        self.vm_types=[]
        for i in range(self.vm_types_size):
            _type = origin_case_info[3+i].replace('\r\n','')
            _typename = _type.split(' ')[0]
            self.vm_types.append(_typename)
        # 处理优化目标    
        self.opt_target = origin_case_info[4+tsize].replace('\r\n','')
        # 处理时间
        _st = origin_case_info[6 + tsize].replace('\r\n','')
        _et = origin_case_info[7+tsize].replace('\n','')
        # 打印起始时间
        print _st,_et
        st = datetime.strptime(_st,"%Y-%m-%d %H:%M:%S")
        et = datetime.strptime(_et,'%Y-%m-%d %H:%M:%S')
        td = et-st
        if predict_time_grain == ParamInfo.TIME_GRAIN_DAY:
            self.date_range_size = td.days
        elif predict_time_grain == ParamInfo.TIME_GRAIN_HOUR:
            self.date_range_size = td.days*24 + td.seconds/ 3600 
        else:
            self.date_range_size = td.days
        self.data_range=[_st,_et]
    
    def set_his_data(self,origin_train_data,predict_time_grain):
        if (origin_train_data is None) or \
            len(origin_train_data) ==0 :
            raise ValueError('Error origin_train_data=',origin_train_data)
        hisdata = {}
        
        for line in origin_train_data:
            line = line.replace('\r\n','')
            _,vmtype,time=line.split('\t')
            if not isContainKey(hisdata, vmtype):
                hisdata[vmtype]={}
            gt = get_grain_time(time,predict_time_grain)
            point = hisdata[vmtype]
            if not isContainKey(point,gt):
                point[gt]=0
            cot = point[gt]+1
            point[gt]=cot
        self.his_data = hisdata

    def add_his_data(self,origin_train_data):
        '''
        在原时间粒度下，添加历史数据信息
        '''
        if (origin_train_data is None) or \
            len(origin_train_data) ==0 :
            raise ValueError('Error origin_train_data=',origin_train_data)
        hisdata=self.his_data
        for line in origin_train_data:
            line = line.replace('\r\n','')
            _,vmtype,time=line.split('\t')
            if not isContainKey(hisdata, vmtype):
                hisdata[vmtype]={}
            gt = get_grain_time(time,self.time_grain)
            point = hisdata[vmtype]
            if not isContainKey(point,gt):
                point[gt]=0
            cot = point[gt]+1
            point[gt]=cot
        self.his_data = hisdata
     
    def get_his_data_by_vmtype(self,vmtype):
        '''
        返回一个从第一个数据时间到预测开始前的数据统计列表
        使用0填补空缺值
        ['time':[时间标签],
        'value':[值]]
        '''
        # TODO 2018-03-21 modify：answer exit abnormal Missing output file
        if  vmtype not in self.his_data:
            result = {'time': [0],  # 时间标签
                      'value': [0]}  # 统计值
            return result
        else:
            result = {'time': [],  # 时间标签
                      'value': []}  # 统计值
        tdict = self.his_data[vmtype]
        tkeys = tdict.keys()
        tkeys.sort()

        hrs = 1
        if self.time_grain == ParamInfo.TIME_GRAIN_DAY:
            hrs = 24
        td = timedelta(hours=hrs)
        st = datetime.strptime(tkeys[0],'%Y-%m-%d %H:%M:%S')
        et = datetime.strptime(self.data_range[0],'%Y-%m-%d %H:%M:%S')

        while st<et:
            timestr = st.strftime('%Y-%m-%d %H:%M:%S')
            result['time'].append(timestr)
            if timestr in tkeys:
                result['value'].append(tdict[timestr])
            else:
                result['value'].append(0)
            st = st+td

            
        return result    

    def toInt(self,value,tType=0):
            if tType==0.0:
                return value
            elif tType==1.0:
                return math.ceil(value)
            elif tType==-1.0:
                return math.floor(value)
            elif tType==0.5:
                return round(value)
            
    def get_his_data_by_vmtype_avage_v1(self,vmtype,toInt=0):
        '''
        返回一个从第一个数据时间到预测开始前的数据统计列表
        使用前后最近平均值填补空缺,若后一段的无法平均值 用最近有效值填补
        ['time':[时间标签],
        'value':[值]]
        '''
        # TODO 2018-03-21 modify：answer exit abnormal Missing output file
        if  vmtype not in self.his_data:
            result = {'time': [0],  # 时间标签
                      'value': [0]}  # 统计值
            return result
        else:
            result = {'time': [],  # 时间标签
                      'value': []}  # 统计值
        tdict = self.his_data[vmtype]
        tkeys = tdict.keys()
        tkeys.sort()
        kno_len = len(tkeys)
        kno_s = 0
        kno_e = 0
        kno_s_value = tdict[tkeys[0]]
        kno_e_value = kno_s_value
           

        hrs = 1
        if self.time_grain == ParamInfo.TIME_GRAIN_DAY:
            hrs = 24
        td = timedelta(hours=hrs)
        st = datetime.strptime(tkeys[0],'%Y-%m-%d %H:%M:%S')
        et = datetime.strptime(self.data_range[0],'%Y-%m-%d %H:%M:%S')

        while st<et:
            timestr = st.strftime('%Y-%m-%d %H:%M:%S')
            result['time'].append(timestr)
            if kno_e<0:
                result['value'].append(self.toInt(kno_s_value, toInt))
            elif timestr == tkeys[kno_e]:
                kno_s_value = kno_e_value
                result['value'].append(self.toInt(kno_s_value, toInt))
                kno_e+=1
                if kno_e==kno_len:
                    kno_e=-1
                kno_e_value = tdict[tkeys[kno_e]]
            else:
                kno_s_value = (kno_s_value+kno_e_value)/2.0
                result['value'].append(self.toInt(kno_s_value, toInt))
            st = st+td
            kno_s+=1
            
        return result        
        
    def get_his_data_by_vmtype_avage_v2(self,vmtype,toInt=0):
        '''
        返回一个从第一个数据时间到预测开始前的数据统计列表
        使用前后最近平均值填补空缺,若后一段的无法平均值 
        用最近前一个星期前数据替代，无法替代则使用最后一个
        ['time':[时间标签],
        'value':[值]]
        '''
        #TODO 2018-03-20 modify：answer exit abnormal Missing output file
        if  vmtype not in self.his_data:
            result = {'time': [0],  # 时间标签
                      'value': [0]}  # 统计值
            return result
        else:
            result = {'time': [],  # 时间标签
                      'value': []}  # 统计值
        tdict = self.his_data[vmtype]
        tkeys = tdict.keys()
        tkeys.sort()
        kno_len = len(tkeys)
        kno_s = 0
        kno_e = 0
        kno_s_value = tdict[tkeys[0]]
        kno_e_value = 0

        # 跨度天数
        t_len = 0
        #四舍五入
        decimal_threshold = 0.5
        #缩小倍数
        enlarge_size=1.0
        #对称填充
        pading_pos = 2
        #周末权重
        week_weight=1
        #购物节权重
        shopping_weight = 1

        hrs = 1
        if self.time_grain == ParamInfo.TIME_GRAIN_DAY:
            hrs = 24
        td = timedelta(hours=hrs)
        st = datetime.strptime(tkeys[0],'%Y-%m-%d %H:%M:%S')
        et = datetime.strptime(self.data_range[0],'%Y-%m-%d %H:%M:%S')

        # #存在两个或两个以上的申请记录,初始化时间跨度
        # if kno_len>2:
        #     provious_time = datetime.strptime(tkeys[0], '%Y-%m-%d %H:%M:%S')
        #     last_time = datetime.strptime(tkeys[1], '%Y-%m-%d %H:%M:%S')
        #     t_len = (provious_time - last_time).days
        #

        while st<et:
            timestr = st.strftime('%Y-%m-%d %H:%M:%S')
            result['time'].append(timestr)
            if kno_e<0:
                vset = result['value']
                if len(vset) >= 7:
                    # temp_result=vset[-pading_pos]
                    temp_result = vset[-7]
                    # pading_pos += 2
                else:#用最后一个数据填充
                    temp_result=0
                # 购物节处理
                if str(st) in ParamInfo.shopping_days:
                    temp_result = temp_result * shopping_weight
                    # if temp_result>decimal_threshold:#阈值进位
                    #     temp_result+=1

                result['value'].append(decimal_Process(temp_result,decimal_threshold))

            elif timestr == tkeys[kno_e]:#当前时间(天)是申请过该类型主机
                #正确时间赋值
                result['value'].append(self.toInt(tdict[tkeys[kno_e]], toInt))
                #指向下一个时间节点
                kno_e+=1
                if kno_e==kno_len:
                    kno_e=-1
                else:#如果后面还有申请记录
                    kno_s_value = kno_e_value
                    kno_e_value = tdict[tkeys[kno_e]]
                    provious_time = datetime.strptime(tkeys[kno_e], '%Y-%m-%d %H:%M:%S')
                    last_time = datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S')
                    t_len=float((provious_time-last_time).days+1)
            else:#当前时间(天),没有申请过该类型主机 TODO BUG
                temp_result = (kno_s_value+kno_e_value)/(t_len*enlarge_size)
                #周六或者周末
                if st.timetuple().tm_wday==5 or st.timetuple().tm_wday==6:
                    temp_result=temp_result/week_weight

                #购物节
                if str(st) in ParamInfo.shopping_days:
                    temp_result = temp_result*shopping_weight

                result['value'].append(decimal_Process(temp_result,decimal_threshold))
                #self.toInt(kno_s_value, toInt)
                # result['value'].append(temp_result)
            t_len-=1
            st = st+td
            kno_s+=1
        #购物节数据放大
        # result=holiday_Process(result,shopping_weight)
        return result


    def get_his_data_by_vmtype_avage_v3(self, vmtype, toInt=0):
        '''
        返回一个真实的时间表
        ['time':[时间标签],
        'value':[值]]
        '''
        #如果该类型并没有出现过，则返回0
        if vmtype not in self.his_data:
            result = {'time': [0],  # 时间标签
                      'value': [0]}  # 统计值
            return result
        else:
            result = {'time': [],  # 时间标签
                      'value': []}  # 统计值
        tdict = self.his_data[vmtype]
        tkeys = tdict.keys()
        tkeys.sort()
        kno_len = len(tkeys)
        kno_pos = 0
        kno_s_value = tdict[tkeys[0]]
        kno_e_value = 0

        hrs = 1
        if self.time_grain == ParamInfo.TIME_GRAIN_DAY:
            hrs = 24
        td = timedelta(hours=hrs)
        st = datetime.strptime(tkeys[0], '%Y-%m-%d %H:%M:%S')
        et = datetime.strptime(self.data_range[0], '%Y-%m-%d %H:%M:%S')
        while st < et:
            timestr = st.strftime('%Y-%m-%d %H:%M:%S')
            result['time'].append(timestr)
            if kno_len != kno_pos:#如果申请表中还有时间
                if timestr == tkeys[kno_pos]:
                    # 正确时间赋值
                    result['value'].append(self.toInt(tdict[tkeys[kno_pos]], toInt))
                    #指向申请时间点
                    kno_pos+=1
                else:
                    result['value'].append(self.toInt(0, toInt))
            else:#没有的就是申请数为0的
                result['value'].append(self.toInt(0,toInt))
            st = st + td
        # 购物节数据放大
        return result


    # def set_feature_map(self):
    #     '''
    #     提取特征,训练模型 34列
    #     '''
    #     self.feature_list = feature_merge(self.vm_types, self.his_data)  # 提取特征
    #
    # def set_predictor_map(self):
    #     '''
    #     获取预测特征表 33列
    #     '''
    #     self.predictor_list = get_predictor_map(self.data_range[0],self.data_range[1],self.vm_types_size)

################### class CaseInfo end #############################


# 获取粒度时间
split_append_tmp=[[13,':00:00'],[10,' 00:00:00']]
def get_grain_time(time_str,time_grain):
    sp_len_tmp = split_append_tmp[time_grain][0]
    sp_str_tmp = split_append_tmp[time_grain][1]
    return time_str[:sp_len_tmp]+sp_str_tmp

# 检查dict中是否存在key
def isContainKey(dic,key):
    return key in dic.keys()

#小数舍入
def decimal_Process(value,decimal_threshold):
    if value==0:
        return 0
    if value%1>decimal_threshold:
        return int(value+1)
    else:
        return int(value)

def holiday_Process(result,shopping_weight):

    time_list = copy.deepcopy(result['time'])
    vlues_list =copy.deepcopy(result['value'])

    for i in range(len(time_list)):
        if time_list[i] in ParamInfo.shopping_days:
            vlues_list[i]*=shopping_weight
    result = {'time': time_list,  # 时间标签
              'value': vlues_list}  # 统计值
    return result



