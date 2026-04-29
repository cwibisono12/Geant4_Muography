#!/usr/bin/env python3

from sklearn.neural_network import MLPRegressor

def get_parameters(X_train, y_train):
    '''
    Get the parameters of the Supervised NN based on Multi-Layer Perceptron
    C. Wibisono
    04/29 '26
    Parameter(s):
    X_train: [arr] independent features variables
    y_train: [arr] target
    Return(s):
    coeff: [obj] parameters of the model
    '''

    coeff  = MLPRegressor(solver = 'lbfgs', alpha = 1e-5, 
            hidden_layer_sizes = (100,10), activation='relu', max_iter = 2000)

    coeff.fit(X_train, y_train)

    return coeff


def predict_outcome(X_test, model_param):
    '''
    Get the result of the model based on the provided model_parameters
    C. Wibisono
    04/29 '26
    Parameter(s):
    X_test: [arr] independent features variables
    model_param: [obj] param of model
    Return(s):
    y_test: [arr] results of imaging
    '''

    y_test = model_param.predict(X_test)

    return y_test


def get_features(file_in):
    '''
    Extract the features data to be used for the model
    C. Wibisono
    04/29 '26
    Parameter(s):
    file_in: fileinput pointer object
    Return(s):
    X: [arr] array of independent features variables
    y: [arr] array of dependent variables
    '''

    with open(file_in, mode='r') as fin:
        X_arr = []
        y_arr = []
        while(1):
            line = fin.readline()
            if line == '':
                break
            else:
                row = line.split(',')
                x1_scint = float(row[1])
                y1_scint = float(row[2])
                z1_scint = float(row[3])
                
                x2_scint = float(row[4])
                y2_scint = float(row[5])
                z2_scint = float(row[6])

                x3_scint = float(row[7])
                y3_scint = float(row[8])
                z3_scint = float(row[9])

                x4_scint = float(row[10])
                y4_scint = float(row[11])
                z4_scint = float(row[12])

                x_pipe = float(row[13])
                y_pipe = float(row[14])
                z_pipe = float(row[15])

                x_scale = float(row[16])
                y_scale = float(row[17])
                z_scale = float(row[18])



                X_arr.append([x1_scint,y1_scint,z1_scint,x2_scint,y2_scint,z2_scint,x3_scint,y3_scint,z3_scint,x4_scint,y4_scint,z4_scint])
                y_arr.append([x_pipe,y_pipe,z_pipe])

        
        return X_arr, y_arr

   
def write_features(fout, y_test):
    '''
    Export the result from the model prediction
    C. Wibisono
    04/29 '26
    Parameter(s):
    fout: file output pointer object to store the result 
    y_test: [arr] the prediction variables array from the model
    '''
    dim = len(y_test)
    with open(fout, mode='w') as f:
        for i in range(dim):
            f.write(str(y_test[i][0])+','+str(y_test[i][1])+','+str(y_test[i][2])+'\n')

def plot_results(coords):
    '''
    Plot the results of the coordinates
    C. Wibisono
    04/29 '26
    Parameter(s):
    coords: Spatial coordinates (x,y,z) of the target
    '''

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(coords[:,0],coords[:,1],coords[:,2],c='blue',marker='o',s=0.001)

    plt.show()
