# -*- coding: utf-8 -*-
import copy
from datetime import timedelta
from datetime import datetime
import math

import const_map
from feature_map import first_workdays, weekend_workdays, holidays, shopping_days


class DataObj(object):
    '''
    训练数据对象
    '''
    CPU = 0  # cpu数
    MEM = 0  # 内存存储量 单位Gb
    HDD = 0  # 硬盘存储量 单位Gb

    mp_type_name = []
    mp_type_list = {}

    opt_target = 'MEM'  # 优化目标，值为CPU和MEM

    vm_types_size = 0  # 虚拟机类型数
    vm_types = []  # 虚拟机类型{list}
    vm_types_count = {}

    time_grain = -1  # 预测时间粒度
    date_range_size = 0  # 需要预测多少天数的数据
    data_range = []  # 预测开始时间与结束时间，左闭右开

    # 预测时间与训练集间隔时间
    gap_time = 0

    # 训练数据中虚拟机
    his_data = {}

    # 训练集 开始-结束 时间
    train_data_range = []
    train_day_count = 0.0  # 训练天数

    # 训练特征表
    train_X = []
    train_Y = []

    # 测试集各种种类的数量
    test_vm_types_count = {}
    test_count = []

    # 测试表
    predictor_data = {}  # {vm:predictor_data}

    def __init__(self, origin_case_info, origin_train_data, predict_time_grain=const_map.TIME_GRAIN_DAY,
                 input_test_file_array=None):
        '''
        初始化DataObj
        '''
        self.time_grain = predict_time_grain
        self.set_input_config(origin_case_info, predict_time_grain)
        self.set_his_data(origin_train_data, predict_time_grain)
        # 提取特征
        # self.set_train_feature()
        # self.set_predictor_feature()
        if input_test_file_array != None:
            self.set_test_list(input_test_file_array)
        # self.set_feature_map()
        # self.set_predictor_map()
        pass

    def set_input_config(self, origin_case_info, predict_time_grain):
        '''
        :param origin_case_info:
        :param predict_time_grain:
        :return:
        '''
        if (origin_case_info is None) or \
                len(origin_case_info) < 2:
            raise ValueError('Error origin_case_info=', origin_case_info)

        # 处理 CPU MEM HDD
        tmp = origin_case_info[0].replace('\r\n', '')
        tmps = tmp.split(' ')
        pm_size = int(tmps[0])
        for i in range(1, pm_size + 1):
            mp_list = {}
            tmp = origin_case_info[i].replace('\r\n', '')
            tmps = tmp.split(' ')
            mp_list['CPU'] = int(tmps[1])
            mp_list['MEM'] = int(tmps[2])
            mp_list['HDD'] = int(tmps[3])
            self.mp_type_list[tmps[0][0]] = mp_list
            self.mp_type_name.append(tmps[0])

        # 处理虚拟机类型
        vm_size = int(origin_case_info[5].replace('\r\n', ''))
        self.vm_types_size = vm_size
        self.vm_types = []
        for i in range(self.vm_types_size):
            _type = origin_case_info[6 + i].replace('\r\n', '')
            _typename = _type.split(' ')[0]
            self.vm_types.append(_typename)
            self.vm_types_count[_typename] = 0
            self.predictor_data[_typename] = []

        # 处理时间
        # _st = origin_case_info[6 + vm_size].replace('\r\n', '')
        # _et = origin_case_info[7 + vm_size].replace('\n', '')
        _st = origin_case_info[4 + pm_size + vm_size][0:19]
        _et = origin_case_info[5 + pm_size + vm_size][0:19]
        # 打印起始时间
        print _st, _et
        st = datetime.strptime(_st, "%Y-%m-%d %H:%M:%S")
        et = datetime.strptime(_et, '%Y-%m-%d %H:%M:%S')

        ts = timedelta(seconds=1)
        # 时间修正+s
        if et.timetuple().tm_hour == 23 and et.timetuple().tm_min == 59 and et.timetuple().tm_sec == 59:
            et += ts
        td = et - st
        if predict_time_grain == const_map.TIME_GRAIN_DAY:
            self.date_range_size = td.days
        elif predict_time_grain == const_map.TIME_GRAIN_HOUR:
            self.date_range_size = td.days * 24 + td.seconds / 3600
        else:
            self.date_range_size = td.days
        self.data_range = [_st, _et]

        self.CPU = self.mp_type_list["G"]["CPU"]
        self.MEM = self.mp_type_list["G"]["MEM"]
        self.HDD = self.mp_type_list["G"]["HDD"]

    def set_his_data(self, origin_train_data, predict_time_grain):
        '''
        :param origin_train_data: 原始数据
        :param predict_time_grain:预测时间范围
        '''
        if (origin_train_data is None) or \
                len(origin_train_data) == 0:
            raise ValueError('Error origin_train_data=', origin_train_data)
        hisdata = {}
        first = 1
        for line in origin_train_data:
            line = line.replace('\r\n', '')
            _, vmtype, time = line.split('\t')
            if not isContainKey(hisdata, vmtype):
                hisdata[vmtype] = {}
            gt = get_grain_time(time, predict_time_grain)
            # 统计训练集中虚拟机出现次数
            if isContainKey(self.vm_types_count, vmtype):
                self.vm_types_count[vmtype] += 1
            # 保存训练集开始时间
            if first == 1:
                self.train_data_range.append(gt)
                first = 0
            point = hisdata[vmtype]
            if not isContainKey(point, gt):
                point[gt] = 0
            cot = point[gt] + 1
            point[gt] = cot
            endtime = time.replace('\n', '')

        # 训练集结束时间
        self.train_data_range.append(get_grain_time(endtime, predict_time_grain))
        end_date = datetime.strptime(self.train_data_range[1], '%Y-%m-%d %H:%M:%S')
        begin_date = datetime.strptime(self.train_data_range[0], '%Y-%m-%d %H:%M:%S')
        # 训练集长度
        self.train_day_count = end_date.timetuple().tm_yday - begin_date.timetuple().tm_yday
        # 计算间隔时间
        end_date = datetime.strptime(self.data_range[0], '%Y-%m-%d %H:%M:%S')
        begin_date = datetime.strptime(endtime, '%Y-%m-%d %H:%M:%S')
        self.gap_time = end_date.timetuple().tm_yday - begin_date.timetuple().tm_yday

        self.his_data = hisdata

        # 测试一台主机

    def add_his_data(self, origin_train_data):
        '''
        添加历史数据信息
        '''
        if (origin_train_data is None) or \
                len(origin_train_data) == 0:
            raise ValueError('Error origin_train_data=', origin_train_data)
        hisdata = self.his_data
        for line in origin_train_data:
            line = line.replace('\r\n', '')
            _, vmtype, time = line.split('\t')
            if not isContainKey(hisdata, vmtype):
                hisdata[vmtype] = {}
            gt = get_grain_time(time, self.time_grain)
            point = hisdata[vmtype]
            if not isContainKey(point, gt):
                point[gt] = 0
            cot = point[gt] + 1
            point[gt] = cot
        self.his_data = hisdata

    def get_his_data_by_vmtype(self, vmtype):
        '''
        返回一个从第一个数据时间到预测开始前的数据统计列表
        使用0填补空缺值
        ['time':[时间标签],
        'value':[值]]
        '''
        # TODO 2018-03-21 modify：answer exit abnormal Missing output file
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

        hrs = 1
        if self.time_grain == const_map.TIME_GRAIN_DAY:
            hrs = 24
        td = timedelta(hours=hrs)
        st = datetime.strptime(tkeys[0], '%Y-%m-%d %H:%M:%S')
        et = datetime.strptime(self.data_range[0], '%Y-%m-%d %H:%M:%S')

        while st < et:
            timestr = st.strftime('%Y-%m-%d %H:%M:%S')
            result['time'].append(timestr)
            if timestr in tkeys:
                result['value'].append(tdict[timestr])
            else:
                result['value'].append(0)
            st = st + td

        return result

    def toInt(self, value, tType=0):
        if tType == 0.0:
            return value
        elif tType == 1.0:
            return math.ceil(value)
        elif tType == -1.0:
            return math.floor(value)
        elif tType == 0.5:
            return round(value)

    def get_data_list_v1(self, vmtype, toInt=0):
        '''
        返回一个从第一个数据时间到预测开始前的数据统计列表
        使用前后最近平均值填补空缺,若后一段的无法平均值 用最近有效值填补
        ['time':[时间标签],
        'value':[值]]
        '''
        # TODO 2018-03-21 modify：answer exit abnormal Missing output file
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
        kno_s = 0
        kno_e = 0
        kno_s_value = tdict[tkeys[0]]
        kno_e_value = kno_s_value

        hrs = 1
        if self.time_grain == const_map.TIME_GRAIN_DAY:
            hrs = 24
        td = timedelta(hours=hrs)
        st = datetime.strptime(tkeys[0], '%Y-%m-%d %H:%M:%S')
        et = datetime.strptime(self.data_range[0], '%Y-%m-%d %H:%M:%S')

        while st < et:
            timestr = st.strftime('%Y-%m-%d %H:%M:%S')
            result['time'].append(timestr)
            if kno_e < 0:
                result['value'].append(self.toInt(kno_s_value, toInt))
            elif timestr == tkeys[kno_e]:
                kno_s_value = kno_e_value
                result['value'].append(self.toInt(kno_s_value, toInt))
                kno_e += 1
                if kno_e == kno_len:
                    kno_e = -1
                kno_e_value = tdict[tkeys[kno_e]]
            else:
                kno_s_value = (kno_s_value + kno_e_value) / 2.0
                result['value'].append(self.toInt(kno_s_value, toInt))
            st = st + td
            kno_s += 1

        return result

    def get_data_list_v2(self, vmtype, toInt=0):
        '''
        返回一个从第一个数据时间到预测开始前的数据统计列表
        使用前后最近平均值填补空缺,若后一段的无法平均值 
        用最近前一个星期前数据替代，无法替代则使用最后一个
        ['time':[时间标签],
        'value':[值]]
        '''
        # TODO 2018-03-20 modify：answer exit abnormal Missing output file
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
        kno_s = 0
        kno_e = 0
        kno_s_value = tdict[tkeys[0]]
        kno_e_value = 0

        # 跨度天数
        t_len = 0
        # 四舍五入
        decimal_threshold = 0.5
        # 缩小倍数
        enlarge_size = 1.0
        # 对称填充
        pading_pos = 2
        # 周末权重
        week_weight = 1
        # 购物节权重
        shopping_weight = 1

        hrs = 1
        if self.time_grain == const_map.TIME_GRAIN_DAY:
            hrs = 24
        td = timedelta(hours=hrs)
        st = datetime.strptime(tkeys[0], '%Y-%m-%d %H:%M:%S')
        et = datetime.strptime(self.data_range[0], '%Y-%m-%d %H:%M:%S')

        # #存在两个或两个以上的申请记录,初始化时间跨度
        # if kno_len>2:
        #     provious_time = datetime.strptime(tkeys[0], '%Y-%m-%d %H:%M:%S')
        #     last_time = datetime.strptime(tkeys[1], '%Y-%m-%d %H:%M:%S')
        #     t_len = (provious_time - last_time).days
        #

        while st < et:
            timestr = st.strftime('%Y-%m-%d %H:%M:%S')
            result['time'].append(timestr)
            if kno_e < 0:
                vset = result['value']
                if len(vset) >= 7:
                    # temp_result=vset[-pading_pos]
                    temp_result = vset[-7]
                    # pading_pos += 2
                else:  # 用最后一个数据填充
                    temp_result = 0
                # 购物节处理
                if str(st) in const_map.shopping_days:
                    temp_result = temp_result * shopping_weight
                    # if temp_result>decimal_threshold:#阈值进位
                    #     temp_result+=1

                result['value'].append(decimal_Process(temp_result, decimal_threshold))

            elif timestr == tkeys[kno_e]:  # 当前时间(天)是申请过该类型主机
                # 正确时间赋值
                result['value'].append(self.toInt(tdict[tkeys[kno_e]], toInt))
                # 指向下一个时间节点
                kno_e += 1
                if kno_e == kno_len:
                    kno_e = -1
                else:  # 如果后面还有申请记录
                    kno_s_value = kno_e_value
                    kno_e_value = tdict[tkeys[kno_e]]
                    provious_time = datetime.strptime(tkeys[kno_e], '%Y-%m-%d %H:%M:%S')
                    last_time = datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S')
                    t_len = float((provious_time - last_time).days + 1)
            else:  # 当前时间(天),没有申请过该类型主机 TODO BUG
                temp_result = (kno_s_value + kno_e_value) / (t_len * enlarge_size)
                # 周六或者周末
                if st.timetuple().tm_wday == 5 or st.timetuple().tm_wday == 6:
                    temp_result = temp_result / week_weight

                # 购物节
                if str(st) in const_map.shopping_days:
                    temp_result = temp_result * shopping_weight

                result['value'].append(decimal_Process(temp_result, decimal_threshold))
                # self.toInt(kno_s_value, toInt)
                # result['value'].append(temp_result)
            t_len -= 1
            st = st + td
            kno_s += 1
        # 购物节数据放大
        # result=holiday_Process(result,shopping_weight)
        return result

    def get_data_list_v3(self, vmtype, toInt=0):
        '''
        返回一个真实的时间表
        ['time':[时间标签],
        'value':[值]]
        '''
        # 如果该类型并没有出现过，则返回0
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
        if self.time_grain == const_map.TIME_GRAIN_DAY:
            hrs = 24
        td = timedelta(hours=hrs)
        st = datetime.strptime(tkeys[0], '%Y-%m-%d %H:%M:%S')
        et = datetime.strptime(self.data_range[0], '%Y-%m-%d %H:%M:%S')
        while st < et:
            timestr = st.strftime('%Y-%m-%d %H:%M:%S')
            result['time'].append(timestr)
            if kno_len != kno_pos:  # 如果申请表中还有时间
                if timestr == tkeys[kno_pos]:
                    # 正确时间赋值
                    result['value'].append(self.toInt(tdict[tkeys[kno_pos]], toInt))
                    # 指向申请时间点
                    kno_pos += 1
                else:
                    result['value'].append(self.toInt(0, toInt))
            else:  # 没有的就是申请数为0的
                result['value'].append(self.toInt(0, toInt))
            st = st + td
        # 购物节数据放大
        return result

    def get_data_list_v4(self, vmtype, toInt=0):
        '''
        返回一个真实的时间表+滤波去噪
        以周为周期,滤波去噪
        '''
        # 如果该类型并没有出现过，则返回0
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
        if self.time_grain == const_map.TIME_GRAIN_DAY:
            hrs = 24
        td = timedelta(hours=hrs)
        st = datetime.strptime(tkeys[0], '%Y-%m-%d %H:%M:%S')
        et = datetime.strptime(self.data_range[0], '%Y-%m-%d %H:%M:%S')
        while st < et:
            timestr = st.strftime('%Y-%m-%d %H:%M:%S')
            result['time'].append(timestr)
            if kno_len != kno_pos:  # 如果申请表中还有时间
                if timestr == tkeys[kno_pos]:
                    # 正确时间赋值
                    result['value'].append(self.toInt(tdict[tkeys[kno_pos]], toInt))
                    # 指向申请时间点
                    kno_pos += 1
                else:
                    result['value'].append(self.toInt(0, toInt))
            else:  # 没有的就是申请数为0的
                result['value'].append(self.toInt(0, toInt))
            st = st + td
        # 获得基本数据集

        ########################原始数据矩阵############################
        orign_martix_data = []
        # 获取平均值
        avg_count = self.vm_types_count[vmtype] / float(self.train_day_count)
        orign_martix_data = reshape_data(result, avg_count)

        ########################原始数据矩阵############################

        # 过滤 filter='average'   filter='gaussian'高斯滤波
        result['value'] = to_filter(orign_martix_data, filter='gaussian', sigma=1, orgin_data_size=self.train_day_count,
                                    avg_count=avg_count)

        return result

    def get_data_list(self, vmtype, toInt, vmtype_avage_v):
        '''
        :param vmtype:虚拟机类型
        :param toInt: 取整类型
        :param vmtype_avage_v:填充方案
        :return: 返回数据列表
        '''
        if vmtype_avage_v == 1:
            return self.get_data_list_v1(vmtype, toInt)
        elif vmtype_avage_v == 2:
            return self.get_data_list_v2(vmtype, toInt)
        elif vmtype_avage_v == 3:
            return self.get_data_list_v3(vmtype, toInt)
        elif vmtype_avage_v == 4:
            return self.get_data_list_v4(vmtype, toInt)

    def get_train_X(self):
        '''
        提取特征,训练模型 27列
        '''
        return self.train_X[0:-1]

    def get_train_Y(self):
        '''
        提取特征,训练模型 7列
        '''
        return self.train_Y

    def get_predictor_X(self, vm_type):
        '''
        获取预测特征表 27列
        :param vm_type: 虚拟机类型
        :return:预测集合
        '''
        return self.predictor_data[vm_type]

    def set_train_feature(self):
        '''
        初始化训练特征集合
        '''
        if self.time_grain == const_map.TIME_GRAIN_DAY:
            hrs = 24
        td = timedelta(hours=hrs)
        st = datetime.strptime(self.train_data_range[0], '%Y-%m-%d %H:%M:%S')
        et = datetime.strptime(self.train_data_range[1], '%Y-%m-%d %H:%M:%S')
        # 遍历时间
        while st < et:
            # 虚拟机特征
            vm_feature = [0] * self.vm_types_size
            # 遍历要预测的虚拟机 当天需要观察的虚拟机(每天/每种虚拟机)
            for i in range(self.vm_types_size):
                # 如果要预测的虚拟机出现过
                if isContainKey(self.his_data, self.vm_types[i]):
                    # 当前时间出现过
                    if isContainKey(self.his_data[self.vm_types[i]], st.strftime("%Y-%m-%d %H:%M:%S")):
                        vm_feature[i] = self.his_data[self.vm_types[i]][st.strftime("%Y-%m-%d %H:%M:%S")]
                    # # 添加出现频率
                    # vm_feature.append(self.vm_types_count[self.vm_types[i]])
                # else:  # 如果要预测的虚拟机没出现过,频率为0
                #     vm_feature.append(0)
            time_fear = self.get_time_feature(st)
            vm_feature.extend(time_fear)
            # 添加如训练集队列
            self.train_X.append(vm_feature)
            st += td

    def set_predictor_feature(self):
        self.train_Y = [a[0:self.vm_types_size] for a in self.train_X[1:]]
        # '''
        # 初始化预测特征集合
        # '''
        # if self.time_grain == ParamInfo.TIME_GRAIN_DAY:
        #     hrs = 24
        # td = timedelta(hours=hrs)
        # st = datetime.strptime(self.data_range[0], '%Y-%m-%d %H:%M:%S')
        # et = datetime.strptime(self.data_range[1], '%Y-%m-%d %H:%M:%S')
        # # 遍历时间
        # while st < et:
        #     # 遍历要预测的虚拟机 当天需要观察的虚拟机(每天/每种虚拟机)
        #     for i in range(self.vm_types_size):
        #         # 虚拟机特征
        #         vm_feature = [0] * self.vm_types_size
        #         vm_feature[i] = 1
        #         # 如果要预测的虚拟机出现过
        #         if isContainKey(self.his_data, self.vm_types[i]):
        #             # 添加出现频率
        #             vm_feature.append(self.vm_types_count[self.vm_types[i]])
        #         else:  # 如果要预测的虚拟机没出现过,频率为0
        #             vm_feature.append(0)
        #         time_fear = self.get_time_feature(st)
        #         vm_feature.extend(time_fear)
        #
        #         # 添加如训练集队列
        #         self.predictor_data[self.vm_types[i]].append(vm_feature)
        #     st += td

    def get_time_feature(self, time):
        '''
        :param time:时间
        :return: 返回时间特征
        '''
        time_feature = []
        # 获取年份
        # year = time.timetuple().tm_year - 2015
        # 获取月份
        # mother = time.timetuple().tm_mon
        # 获取月日
        # day_of_mother = time.timetuple().tm_mday
        # 获得星期几
        dayofweek = time.timetuple().tm_wday + 1
        # 获得第几天数
        # dayofyear = time.timetuple().tm_yday
        # 获得第几周
        weekofyear = int(time.strftime("%W")) + 1
        # 日期
        date_str = time.strftime("%Y-%m-%d")
        # 比较
        if date_str in first_workdays:  # 国家规定的第一个上班日
            first_workday = 1
        else:
            first_workday = 0

        if date_str in weekend_workdays:  # 国家规定周末上班日
            weekend_overtime = 1
        else:
            weekend_overtime = 0

        if date_str in holidays:  # 国家规定假期加班日
            holiday = 1
        else:
            holiday = 0

        # if date_str in shopping_days:  # 购物节
        #     shopping_day = 1
        # else:
        #     shopping_day = 0

        if dayofweek == 6 or dayofweek == 7:  # 周六周末
            weekend = 1
        else:
            weekend = 0
        # time_feature.extend(
        #     [year, mother, day_of_mother, dayofweek, dayofyear, weekofyear, first_workday, weekend_overtime, holiday,
        #      shopping_day, weekend])

        time_feature.extend(
            [dayofweek, weekofyear, first_workday, weekend_overtime, holiday,
             weekend])

        return time_feature

    def set_test_list(self, origin_test_data):
        for line in origin_test_data:
            line = line.replace('\r\n', '')
            _, vmtype, time = line.split('\t')
            # 如果不存在,则初始化
            if not isContainKey(self.test_vm_types_count, vmtype):
                self.test_vm_types_count[vmtype] = 1
            else:  # 存在则计数增加
                self.test_vm_types_count[vmtype] += 1

        for vm_type in self.vm_types:
            if isContainKey(self.test_vm_types_count, vm_type):
                self.test_count.append(self.test_vm_types_count[vm_type])
            else:
                self.test_count.append(0)

    def get_test_list(self):
        return self.test_count


################### class CaseInfo end #############################


split_append_tmp = [[13, ':00:00'], [10, ' 00:00:00']]


def reshape_data(result, avg_count):
    '''
    :param result:历史数据
    :param avg_count: 平均值
    :return: 填充整形厚的数据
    '''
    # 平铺
    martix_data = []
    data_len = len(result['value'])
    orign_martix_data = []

    # 原始数据映射表
    count_map = result['value']

    # 周数
    week_count = len(count_map) / 7
    # 周数
    for w in range(week_count):
        start = 7 * w
        end = 7 * (w + 1)
        # 填充数据
        orign_martix_data.append(count_map[start:end])

    # 获取最后一行数据
    temp = count_map[(week_count) * 7:]

    # 填充平均值
    while (len(temp) != 7):
        temp.append(avg_count)

    # 填充最后一行数据
    orign_martix_data.append(temp)

    return orign_martix_data


def to_filter(orign_martix_data, filter='gaussian', sigma=1, orgin_data_size=0, avg_count=0):
    '''
    :param martix_data:需要过滤的数据
    :param filter: 滤波类型
    :param sigma: 窗口大小
    '''
    if filter == 'average':
        kernel = [[0.0] * (sigma * 2 + 1) for x in range(sigma * 2 + 1)]
        for x in range(-sigma, sigma + 1):
            for y in range(-sigma, sigma + 1):
                kernel[x][y]=1.0
        # 填充padding
        martix_data = to_pading(orign_martix_data, kernel, padding_data=avg_count)

        return averageFilter(martix_data, kernel, orgin_data_size)

    if filter == 'gaussian':
        kernel = [[0.0] * (sigma * 2 + 1) for x in range(sigma * 2 + 1)]
        for x in range(-sigma, sigma + 1):
            for y in range(-sigma, sigma + 1):
                kernel[x + sigma][y + sigma] = math.exp(-0.5 * (x ** 2 + y ** 2) / (sigma ** 2))

        # 填充padding
        martix_data = to_pading(orign_martix_data, kernel, padding_data=avg_count)

        return averageFilter(martix_data, kernel, orgin_data_size)


def averageFilter(martix_data, kernel, orgin_data_size):
    '''
    :param martix_data:数据矩阵
    :param kernel: 核
    :param orgin_data_size:原始数据长度
    :return:
    '''
    return dataConvolv(martix_data, kernel, orgin_data_size)


def dataConvolv(martix_data, kernel, orgin_data_size):
    '''
    :param martix_data:数据矩阵
    :param kernel: 核
    :param orgin_data_size:原始数据长度
    :return:
    '''
    kernel_h = len(kernel)
    kernel_w = len(kernel[0])
    col = len(martix_data[0])
    row = len(martix_data)
    kernel_sum = 0.0
    # 求和
    for i in range(kernel_h):
        for j in range(kernel_w):
            kernel_sum += kernel[i][j]

    kernel_h = kernel_h / 2
    kernel_w = kernel_w / 2
    data_len = orgin_data_size
    avg_data = []
    temp = 0.0
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            temp = 0.0
            # 单个ceil
            for h in range(-kernel_h, kernel_h + 1):  # [-1,1]
                for w in range(-kernel_w, kernel_w + 1):  # [-1,1]
                    temp += martix_data[i + h][j + w] * kernel[h + kernel_h][w + kernel_w]
            temp /= kernel_sum
            avg_data.append(temp)
            # 有效数据卷积完毕
            data_len -= 1
            if data_len == 0:
                break
    return avg_data


def to_pading(orign_martix_data, kernel, padding_data):
    '''

    :param orign_martix_data:原始矩阵数据
    :param kernel: 滤波窗口
    :param padding_data: 填充数据
    :return: 返回填充完的数据
    '''
    kernel_h = len(kernel) / 2
    kernel_w = len(kernel[0]) / 2

    col = len(orign_martix_data[0])
    row = len(orign_martix_data)
    martix_data = copy.deepcopy(orign_martix_data)

    padding_row = []
    # pading行的item个数
    for i in range(col + 2 * kernel_w):
        padding_row.append(padding_data)

    for h in range(kernel_h):  # 上部填充
        martix_data.insert(0, padding_row)
    for w in range(kernel_w):  # 下部填充
        martix_data.insert(row + 1, padding_row)

    temp = []
    # 中间数据填充首尾部
    for i in range(kernel_h, row + kernel_h):
        temp = []
        # 首部插入
        temp = martix_data[i - 1][-kernel_w * 2:-kernel_w]
        # 原始数据插入
        temp.extend(martix_data[i])
        # 尾部数据插入
        temp.extend(martix_data[i + 1][0:kernel_w])
        martix_data[i] = temp
        # martix_data[i].insert(0, martix_data[i-1][-kernel_w*2:-kernel_w])
        # # 尾部插入
        # martix_data[i].insert(1 + col, martix_data[i+1][0:kernel_w])
    return martix_data


def get_grain_time(time_str, time_grain):
    sp_len_tmp = split_append_tmp[time_grain][0]
    sp_str_tmp = split_append_tmp[time_grain][1]
    return time_str[:sp_len_tmp] + sp_str_tmp


def isContainKey(dic, key):
    return key in dic


def decimal_Process(value, decimal_threshold):
    if value == 0:
        return 0
    if value % 1 > decimal_threshold:
        return int(value + 1)
    else:
        return int(value)


def holiday_Process(result, shopping_weight):
    time_list = copy.deepcopy(result['time'])
    vlues_list = copy.deepcopy(result['value'])

    for i in range(len(time_list)):
        if time_list[i] in const_map.shopping_days:
            vlues_list[i] *= shopping_weight
    result = {'time': time_list,  # 时间标签
              'value': vlues_list}  # 统计值
    return result
