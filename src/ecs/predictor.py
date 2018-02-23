
def predict_vm(ecs_lines, input_lines):
    # Do your work from here#
    result = []
    if ecs_lines is None:
        print 'ecs information is none'
        return result
    if input_lines is None:
        print 'input file information is none'
        return result

    for index, item in ecs_lines:
        values = item.split(" ")
        uuid = values[0]
        flavorName = values[1]
        createTime = values[2]

    for index, item in input_lines:
        print "index of input data"

    return result
