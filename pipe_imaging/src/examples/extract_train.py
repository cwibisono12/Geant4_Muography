#!/usr/bin/env python3


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    import nn

    '''
    Below is an example how to use module for extracting training data set.
    '''

    file_train = sys.argv[1]
    option = int(sys.argv[2])
    
    temp = file_train.split('/')
    dim = len(temp)
    f_name = ''

    for i in range(dim):
        if '.csv' in temp[i]:
            f_name = f_name + temp[i].split('new.csv')[0]


    print(f_name)
    #print(fname[2].split('.')[0])

    X_train, y_train = nn.get_features_append(file_train, arg_number = option)
    nn.write_features('../../corr_files/'+str(f_name)+str(option)+'_train.csv', y_train)

    

    

    
