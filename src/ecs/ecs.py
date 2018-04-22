# coding=utf-8
import sys
import os

import predict

# python2 ecs.py data/TrainData_2015.1.1_2015.2.19.txt  data/input_5flavors_cpu_7days.txt result/output1.txt


is_Dubug = False


def main():
    if is_Dubug:
        # #设置路径
        # path = 'data/continuous'
        path='data'
        ecsDataPath = path + '/TrainData_2015.12.txt'
        inputFilePath = path + '/input_3hosttypes_5flavors_1week.txt'
        resultFilePath = 'result/output0.txt'
    else:
        print 'main function begin.'
        if len(sys.argv) != 4:
            print 'parameter is incorrect!'
            print 'Usage: python esc.py ecsDataPath inputFilePath resultFilePath'
            exit(1)
        ecsDataPath = sys.argv[1]
        inputFilePath = sys.argv[2]
        resultFilePath = sys.argv[3]

    # 获取训练集列表
    ecs_infor_array = read_lines(ecsDataPath)
    # 获取输入配置列表
    input_file_array = read_lines(inputFilePath)

    # 预测 Step 01
    if is_Dubug:
        testFilePath = 'data/TestData_2016.1.8_2016.1.14.txt'
        input_test_file_array = read_lines(testFilePath)
        predic_result = predict.predict_vm(ecs_infor_array, input_file_array, input_test_file_array)
    else:
        predic_result = predict.predict_vm(ecs_infor_array, input_file_array, None)
    # 写入结果到文件
    if len(predic_result) != 0:
        write_result(predic_result, resultFilePath)
    else:
        predic_result.append("NA")
        write_result(predic_result, resultFilePath)
    print 'main function end.'


def write_result(array, outpuFilePath):
    with open(outpuFilePath, 'w') as output_file:
        for item in array:
            output_file.write("%s\n" % item)


def read_lines(file_path):
    if os.path.exists(file_path):
        array = []
        with open(file_path, 'r') as lines:
            for line in lines:
                array.append(line)
        return array
    else:
        print 'file not exist: ' + file_path
        return None


if __name__ == "__main__":
    main()
