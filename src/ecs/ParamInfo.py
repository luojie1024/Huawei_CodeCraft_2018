# -*- coding: utf-8 -*-


# import platform

# 所有虚拟机的参数
# [CPU,MEM, W(M/C) ] [U数，M数，存储比核的权重]
VM_PARAM = {
    'flavor1': [1, 1, 1.0],
    'flavor2': [1, 2, 2.0],
    'flavor3': [1, 4, 4.0],

    'flavor4': [2, 2, 1.0],
    'flavor5': [2, 4, 2.0],
    'flavor6': [2, 8, 4.0],

    'flavor7': [4, 4, 1.0],
    'flavor8': [4, 8, 2.0],
    'flavor9': [4, 16, 4.0],

    'flavor10': [8, 8, 1.0],
    'flavor11': [8, 16, 2.0],
    'flavor12': [8, 32, 4.0],

    'flavor13': [16, 16, 1.0],
    'flavor14': [16, 32, 2.0],
    'flavor15': [16, 64, 4.0]
}

VM_CPU_QU = {  # CPU多,内存少的优先,从后往前

    'flavor3': [1, 4, 4.0],
    'flavor6': [2, 8, 4.0],
    'flavor9': [4, 16, 4.0],
    'flavor12': [8, 32, 4.0],
    'flavor15': [16, 64, 4.0],

    'flavor2': [1, 2, 2.0],
    'flavor5': [2, 4, 2.0],
    'flavor8': [4, 8, 2.0],
    'flavor11': [8, 16, 2.0],
    'flavor14': [16, 32, 2.0],

    'flavor1': [1, 1, 1.0],
    'flavor4': [2, 2, 1.0],
    'flavor7': [4, 4, 1.0],
    'flavor10': [8, 8, 1.0],
    'flavor13': [16, 16, 1.0],

}
VM_MEM_QU = {# MEM,CPU少的优先,从后往前
    'flavor1': [1, 1, 1.0],
    'flavor4': [2, 2, 1.0],
    'flavor7': [4, 4, 1.0],
    'flavor10': [8, 8, 1.0],
    'flavor13': [16, 16, 1.0],

    'flavor2': [1, 2, 2.0],
    'flavor5': [2, 4, 2.0],
    'flavor8': [4, 8, 2.0],
    'flavor11': [8, 16, 2.0],
    'flavor14': [16, 32, 2.0],

    'flavor3': [1, 4, 4.0],
    'flavor6': [2, 8, 4.0],
    'flavor9': [4, 16, 4.0],
    'flavor12': [8, 32, 4.0],
    'flavor15': [16, 64, 4.0]
}

VM_TYPE_DIRT = ['flavor1', 'flavor2', 'flavor3', 'flavor4', 'flavor5',
                'flavor6', 'flavor7', 'flavor8', 'flavor9', 'flavor10',
                'flavor11', 'flavor12', 'flavor13', 'flavor14', 'flavor15']

# L1 type_size=3
VM_TYPE_MODIFY1 = {  # flavor1 1 flavor2 12 flavor8 -3
    'flavor1': 2, 'flavor2': 11, 'flavor3': 0, 'flavor4': 0, 'flavor5': 0,
    'flavor6': 0, 'flavor7': 0, 'flavor8': -3, 'flavor9': 0, 'flavor10': 0,
    'flavor11': 0, 'flavor12': 0, 'flavor13': 0, 'flavor14': 0, 'flavor15': 0
}

# L2 type_size=5
VM_TYPE_MODIFY2 = {  # flavor1 3 flavor2 16  flavor5 0  flavor8 -4 flavor9 0
    'flavor1': 3, 'flavor2': 16, 'flavor3': 0, 'flavor4': 0, 'flavor5': 0,
    'flavor6': 0, 'flavor7': 0, 'flavor8': -4, 'flavor9': 0, 'flavor10': 0,
    'flavor11': 0, 'flavor12': 0, 'flavor13': 0, 'flavor14': 0, 'flavor15': 0
}

# L1 type_size=3
VM_TYPE_MODIFY3 = {  # flavor1 2 flavor5 -13 flavor8 6
    'flavor1': 2, 'flavor2': 0, 'flavor3': 0, 'flavor4': 0, 'flavor5': -13,
    'flavor6': 0, 'flavor7': 0, 'flavor8': 6, 'flavor9': 0, 'flavor10': 0,
    'flavor11': 0, 'flavor12': 0, 'flavor13': 0, 'flavor14': 0, 'flavor15': 0
}

# L2 type_size=5
VM_TYPE_MODIFY4 = {# flavor1  flavor2 -12  flavor5 -10 flavor8 8 flavor11  	80.975->81.7
    'flavor1': 0, 'flavor2': -12, 'flavor3':0, 'flavor4': 0, 'flavor5': -10,
    'flavor6': 0, 'flavor7': 0, 'flavor8':12, 'flavor9': 0, 'flavor10': 0,
    'flavor11': -1, 'flavor12': 0, 'flavor13': 0, 'flavor14': 0, 'flavor15': 0
}

# 预测时间粒度
# 
TIME_GRAIN_HOUR = 0
TIME_GRAIN_DAY = 1
TIME_GRAIN_MORE_DAY = 2

holidays = [
    '2013-01-01', '2013-01-02', '2013-01-03',
    '2013-02-09', '2013-02-10', '2013-02-11', '2013-02-12', '2013-02-13', '2013-02-14', '2013-02-15',
    '2013-04-04', '2013-04-05', '2013-04-06',
    '2013-04-29', '2013-04-30', '2013-05-01',
    '2013-06-10', '2013-06-11', '2013-06-12',
    '2013-09-19', '2013-09-20', '2013-09-21',
    '2013-10-01', '2013-10-02', '2013-10-03', '2013-10-04', '2013-10-05', '2013-10-06', '2013-10-07',
    '2014-01-01',
    '2014-01-31', '2014-02-01', '2014-02-02', '2014-02-03', '2014-02-04', '2014-02-05', '2014-02-06',
    '2014-04-05', '2014-04-06', '2014-04-07',
    '2014-05-01', '2014-05-02', '2014-05-03',
    '2014-05-31', '2014-06-01', '2014-06-02',
    '2014-09-06', '2014-09-07', '2014-09-08',
    '2014-10-01', '2014-10-02', '2014-10-03', '2014-10-04', '2014-10-05', '2014-10-06', '2014-10-07',
    '2015-01-01', '2015-01-02', '2015-01-03',
    '2015-02-18', '2015-02-19', '2015-02-20', '2015-02-21', '2015-02-22', '2015-02-23', '2015-02-24',
    '2015-04-05', '2015-04-06',
    '2015-05-01', '2015-05-02', '2015-05-03',
    '2015-06-20', '2015-06-21', '2015-06-22',
    '2015-09-27',
    '2015-10-01', '2015-10-02', '2015-10-03', '2015-10-04', '2015-10-05', '2015-10-06', '2015-10-07',
    '2016-01-01',
    '2016-02-07', '2016-02-08', '2016-02-09', '2016-02-10', '2016-02-11', '2016-02-12', '2016-02-13',
    '2016-04-02', '2016-04-03', '2016-04-04',
    '2016-05-01', '2016-05-02',
    '2016-06-09', '2016-06-10', '2016-06-11',
    '2016-09-15', '2016-09-16', '2016-09-17',
    '2016-10-01', '2016-10-02', '2016-10-03', '2016-10-04', '2016-10-05', '2016-10-06', '2016-10-07',
    '2017-01-01', '2017-01-02',
    '2017-01-27', '2017-01-28', '2017-01-29', '2017-01-30', '2017-01-31', '2017-02-01', '2017-02-02',
    '2017-04-02', '2017-04-03', '2017-04-04',
    '2017-05-01',
    '2017-05-28', '2017-05-29', '2017-05-30',
    '2017-10-01', '2017-10-02', '2017-10-03', '2017-10-04', '2017-10-05', '2017-10-06', '2017-10-07', '2017-10-08',
    '2018-01-01',
]
weekend_workdays = [
    '2013-01-05', '2013-01-06',
    '2013-02-16', '2013-02-17',
    '2013-04-07',
    '2013-04-27', '2013-04-28',
    '2013-06-08', '2013-06-09',
    '2013-09-22',
    '2013-09-29',
    '2013-10-12',
    '2014-01-26', '2014-02-08',
    '2014-05-04',
    '2014-09-28', '2014-10-11',
    '2015-01-04',
    '2015-02-15', '2015-02-28',
    '2015-10-10',
    '2016-02-06', '2016-02-14',
    '2016-06-12',
    '2016-09-18',
    '2016-10-08', '2016-10-09',
    '2017-01-22', '2017-02-04',
    '2017-04-01',
    '2017-05-27',
    '2017-09-30',
]
first_workdays = [
    '2013-01-04',
    '2014-01-06',
    '2014-01-07',
    '2015-01-04',
    '2015-01-05',
    '2015-01-06',
    '2015-01-07',
    '2015-01-08',
    '2016-01-04',
    '2016-01-05',
    '2016-01-06',
    '2017-01-03',
    '2017-01-04',
    '2017-01-05',
    '2017-01-06',
    '2018-01-02',
    '2018-01-03',
    '2018-01-04',
]

# 618[3,2] 11[4,3] 12[4,3]
shopping_days = [

    '2013-06-15 00:00:00', '2013-06-16 00:00:00', '2013-06-17 00:00:00',
    '2013-06-18 00:00:00', '2013-06-19 00:00:00',

    '2013-11-07 00:00:00', '2013-11-08 00:00:00', '2013-11-09 00:00:00', '2013-11-10 00:00:00',
    '2013-11-11 00:00:00', '2013-11-12 00:00:00', '2013-11-13 00:00:00',

    '2013-12-08 00:00:00', '2013-12-09 00:00:00', '2013-12-10 00:00:00', '2013-12-11 00:00:00',
    '2013-12-12 00:00:00', '2013-12-13 00:00:00', '2013-12-14 00:00:00',

    '2014-06-15 00:00:00', '2014-06-16 00:00:00', '2014-06-17 00:00:00',
    '2014-06-18 00:00:00', '2014-06-19 00:00:00',

    '2014-11-07 00:00:00', '2014-11-08 00:00:00', '2014-11-09 00:00:00', '2014-11-10 00:00:00',
    '2014-11-11 00:00:00', '2014-11-12 00:00:00', '2014-11-13 00:00:00',

    '2014-12-08 00:00:00', '2014-12-09 00:00:00', '2014-12-10 00:00:00', '2014-12-11 00:00:00',
    '2014-12-12 00:00:00', '2014-12-13 00:00:00', '2014-12-14 00:00:00',

    # # 元旦
    # '2014-12-31 00:00:00', '2015-01-01 00:00:00',
    #
    # # 元旦
    # '2014-12-31 00:00:00', '2015-01-01 00:00:00',
    #
    # # 春节
    # '2015-02-17 00:00:00', '2015-02-18 00:00:00',
    #
    # # 元宵
    # '2015-03-04 00:00:00', '2015-03-05 00:00:00',
    #
    # # 母亲节
    # '2015-05-10 00:00:00', '2015-05-11 00:00:00',
    #
    # # 儿童节
    # '2015-06-01 00:00:00', '2015-06-02 00:00:00',
    #
    # # 父亲节+端午节
    # '2015-06-19 00:00:00', '2015-06-20 00:00:00', '2015-06-21 00:00:00', '2015-06-22 00:00:00',
    #
    # # 七夕节
    # '2015-08-20 00:00:00', '2015-08-21 00:00:00',
    #
    # # 中秋节
    # '2015-09-26 00:00:00', '2015-09-27 00:00:00', '2015-09-28 00:00:00',
    #
    # # 国庆节
    # '2015-10-01 00:00:00', '2015-10-02 00:00:00',
    #
    # # 万圣节
    # '2015-10-31 00:00:00',
    #
    # # 感恩节
    # '2015-11-26 00:00:00',
    #
    # # 平安夜+圣诞节
    # '2015-12-23 00:00:00', '2015-12-24 00:00:00', '2015-12-25 00:00:00', '2015-12-26 00:00:00',

    '2015-06-16 00:00:00', '2015-06-17 00:00:00',
    '2015-06-18 00:00:00',

    '2015-11-07 00:00:00', '2015-11-08 00:00:00',
    '2015-11-09 00:00:00', '2015-11-10 00:00:00',
    '2015-11-11 00:00:00',
    # '2015-11-12 00:00:00', '2015-11-13 00:00:00',

    # '2015-12-08 00:00:00', '2015-12-09 00:00:00',
    '2015-12-10 00:00:00', '2015-12-11 00:00:00',
    '2015-12-12 00:00:00',
    # '2015-12-13 00:00:00', '2015-12-14 00:00:00',

    '2016-06-15 00:00:00', '2016-06-16 00:00:00',
    '2016-06-17 00:00:00',
    '2016-06-18 00:00:00',
    # '2016-06-19 00:00:00',

    '2016-11-07 00:00:00', '2016-11-08 00:00:00', '2016-11-09 00:00:00', '2016-11-10 00:00:00',
    '2016-11-11 00:00:00',
    '2016-11-12 00:00:00', '2016-11-13 00:00:00',

    '2016-12-08 00:00:00', '2016-12-09 00:00:00', '2016-12-10 00:00:00', '2016-12-11 00:00:00',
    '2016-12-12 00:00:00',
    '2016-12-13 00:00:00', '2016-12-14 00:00:00',

    '2017-06-15 00:00:00', '2017-06-16 00:00:00', '2017-06-17 00:00:00',
    '2017-06-18 00:00:00',
    '2017-06-19 00:00:00',

    '2017-11-07 00:00:00', '2017-11-08 00:00:00', '2017-11-09 00:00:00', '2017-11-10 00:00:00',
    '2017-11-11 00:00:00',
    '2017-11-12 00:00:00', '2017-11-13 00:00:00',

    '2017-12-08 00:00:00', '2017-12-09 00:00:00', '2017-12-10 00:00:00', '2017-12-11 00:00:00',
    '2017-12-12 00:00:00',
    '2017-12-13 00:00:00', '2017-12-14 00:00:00',

]

# # 618[3,2] 11[4,3] 12[4,3]
# shopping_days = [
#
#     '2013-06-15 00:00:00', '2013-06-16 00:00:00', '2013-06-17 00:00:00',
#     '2013-06-18 00:00:00', '2013-06-19 00:00:00',
#
#     '2013-11-07 00:00:00', '2013-11-08 00:00:00', '2013-11-09 00:00:00', '2013-11-10 00:00:00',
#     '2013-11-11 00:00:00', '2013-11-12 00:00:00', '2013-11-13 00:00:00',
#
#     '2013-12-08 00:00:00', '2013-12-09 00:00:00', '2013-12-10 00:00:00', '2013-12-11 00:00:00',
#     '2013-12-12 00:00:00', '2013-12-13 00:00:00', '2013-12-14 00:00:00',
#
#     '2014-06-15 00:00:00', '2014-06-16 00:00:00', '2014-06-17 00:00:00',
#     '2014-06-18 00:00:00', '2014-06-19 00:00:00',
#
#     '2014-11-07 00:00:00', '2014-11-08 00:00:00', '2014-11-09 00:00:00', '2014-11-10 00:00:00',
#     '2014-11-11 00:00:00', '2014-11-12 00:00:00', '2014-11-13 00:00:00',
#
#     '2014-12-08 00:00:00', '2014-12-09 00:00:00', '2014-12-10 00:00:00', '2014-12-11 00:00:00',
#     '2014-12-12 00:00:00', '2014-12-13 00:00:00', '2014-12-14 00:00:00',
#
#     # 元旦
#     '2014-12-31 00:00:00', '2015-01-01 00:00:00',
#
#     # 元旦
#     '2014-12-31 00:00:00', '2015-01-01 00:00:00',
#
#     # 春节
#     '2015-02-17 00:00:00', '2015-02-18 00:00:00',
#
#     # 元宵
#     '2015-03-04 00:00:00', '2015-03-05 00:00:00',
#
#     # 母亲节
#     '2015-05-10 00:00:00', '2015-05-11 00:00:00',
#
#     # 儿童节
#     '2015-06-01 00:00:00', '2015-06-02 00:00:00',
#
#     # 父亲节+端午节
#     '2015-06-19 00:00:00', '2015-06-20 00:00:00', '2015-06-21 00:00:00', '2015-06-22 00:00:00',
#
#     # 七夕节
#     '2015-08-20 00:00:00', '2015-08-21 00:00:00',
#
#     # 中秋节
#     '2015-09-26 00:00:00', '2015-09-27 00:00:00', '2015-09-28 00:00:00',
#
#     # 国庆节
#     '2015-10-01 00:00:00', '2015-10-02 00:00:00',
#
#     # 万圣节
#     '2015-10-31 00:00:00',
#
#     # 感恩节
#     '2015-11-26 00:00:00',
#
#     # 平安夜+圣诞节
#     '2015-12-23 00:00:00', '2015-12-24 00:00:00', '2015-12-25 00:00:00', '2015-12-26 00:00:00',
#
#     '2015-06-16 00:00:00', '2015-06-17 00:00:00',
#     '2015-06-18 00:00:00',
#
#     '2015-11-07 00:00:00', '2015-11-08 00:00:00', '2015-11-09 00:00:00', '2015-11-10 00:00:00',
#     '2015-11-11 00:00:00', '2015-11-12 00:00:00', '2015-11-13 00:00:00',
#
#     '2015-12-08 00:00:00', '2015-12-09 00:00:00', '2015-12-10 00:00:00', '2015-12-11 00:00:00',
#     '2015-12-12 00:00:00', '2015-12-13 00:00:00', '2015-12-14 00:00:00',
#
#     '2016-06-15 00:00:00', '2016-06-16 00:00:00', '2016-06-17 00:00:00',
#     '2016-06-18 00:00:00', '2016-06-19 00:00:00',
#
#     '2016-11-07 00:00:00', '2016-11-08 00:00:00', '2016-11-09 00:00:00', '2016-11-10 00:00:00',
#     '2016-11-11 00:00:00', '2016-11-12 00:00:00', '2016-11-13 00:00:00',
#
#     '2016-12-08 00:00:00', '2016-12-09 00:00:00', '2016-12-10 00:00:00', '2016-12-11 00:00:00',
#     '2016-12-12 00:00:00', '2016-12-13 00:00:00', '2016-12-14 00:00:00',
#
#     '2017-06-15 00:00:00', '2017-06-16 00:00:00', '2017-06-17 00:00:00',
#     '2017-06-18 00:00:00', '2017-06-19 00:00:00',
#
#     '2017-11-07 00:00:00', '2017-11-08 00:00:00', '2017-11-09 00:00:00', '2017-11-10 00:00:00',
#     '2017-11-11 00:00:00', '2017-11-12 00:00:00', '2017-11-13 00:00:00',
#
#     '2017-12-08 00:00:00', '2017-12-09 00:00:00', '2017-12-10 00:00:00', '2017-12-11 00:00:00',
#     '2017-12-12 00:00:00', '2017-12-13 00:00:00', '2017-12-14 00:00:00',
#
# ]
