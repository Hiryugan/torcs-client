import csv
import numpy as np


import csv
import numpy as np



def get_data(files=['alpine-1.csv'], folder='../data/'):

    #files = ['aalborg.csv', 'alpine-1.csv' , 'f-speedway.csv']
    # files = ['aalborg.csv']#, 'alpine-1.csv' , 'f-speedway.csv']

    data = []
    tags = []
    dim = 0
    for fname in files:
        f = open(folder + fname)
        reader = csv.reader(f, delimiter=',')

        i = 0
        for row in reader:
            row = [x for x in row if x != '']

            if i == 0:
                tags = row
                dim = len(row)
                print(dim)
            else:
                if len(row) != dim:
                    print('different number of columns at row {0}: had dim of {1} and got {2} dimensions'.format(i, dim, len(row)))
                else:
                    data.append([float(x) for x in row])
            i += 1

    # npdata = np.zeros((len(data), len(data[0])))
    # for i, row in enumerate(data):
    #     row = list(map(float, row))
    #     print(row)
    #     npdata[i] = np.array(row)
    npdata = np.array(data, dtype=np.float32)
    return npdata, np.array(tags)



def get_data2(files='alpine-1.csv', folder='../data/'):

    #files = ['aalborg.csv', 'alpine-1.csv' , 'f-speedway.csv']
    # files = ['aalborg.csv']#, 'alpine-1.csv' , 'f-speedway.csv']

    data = []
    tags = []
    dim = 0


    f = open(folder + files)

    for row in f:
        datarow = []
        vars = row.split(')')
        j = 0
        for var in vars:
            a = var.split(' ')
            # print(a)
            name = a[0]
            if name == '\n':
                break
            if len(a) > 2 and name != '(opponents':
                for v in a[1:]:
                    datarow.append(v)
            elif name != '(gear2' and name != '(opponents':
                datarow.append(float(a[1]))
        data.append(datarow)

    # npdata = np.zeros((len(data), len(data[0])))
    # for i, row in enumerate(data):
    #     row = list(map(float, row))
    #     print(row)
    #     npdata[i] = np.array(row)
    npdata = np.array(data, dtype=np.float32)

    print("letto shape", npdata.shape)
    return npdata, np.array(tags)
