# coding=utf-8
import sys
import os

import predictor

#python2 ecs.py data/TrainData_2015.1.1_2015.2.19.txt  data/input_5flavors_cpu_7days.txt result/output.txt


input=1
def main():
    if input==2:
        print 'main function begin.'
        if len(sys.argv) != 4:
            print 'parameter is incorrect!'
            print 'Usage: python esc.py ecsDataPath inputFilePath resultFilePath'
            exit(1)
        ecsDataPath = sys.argv[1]
        inputFilePath = sys.argv[2]
        resultFilePath = sys.argv[3]
    else:
        # #设置路径
        case_path = '/home/luojie/桌面/project/log'
        ecsDataPath = case_path + r'/data/TrainData_2015.1.1_2015.2.19.txt'
        inputFilePath = case_path + r'/data/input_5flavors_cpu_7days.txt'
        resultFilePath = case_path + r'/result/result_out.txt'

    #获取训练集列表
    ecs_infor_array = read_lines(ecsDataPath)
    #获取输入配置列表
    input_file_array = read_lines(inputFilePath)

    #预测 Step 01
    predic_result = predictor.predict_vm(ecs_infor_array, input_file_array)
    #写入结果到文件
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
