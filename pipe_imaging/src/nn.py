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


def get_features(file_in, *, arg_number = 3):
    '''
    Extract the features data to be used for the model
    C. Wibisono
    04/29 '26
    Parameter(s):
    file_in: fileinput pointer object
    arg_number: (int) the target variables 1 for pipe only, 2 for scaling only, 3 for both
    Return(s):
    X: [arr] array of independent features variables
    y: [arr] array of dependent variables
    '''

    with open(file_in, mode='r') as fin:
        X_arr = []
        y_arr = []
        #Read header:
        fin.readline()
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
                #y_arr.append([x_pipe,y_pipe,z_pipe])
                if arg_number == 1:
                    y_arr.append([x_pipe,y_pipe,z_pipe])
                if arg_number == 2:
                    y_arr.append([x_scale,y_scale,z_scale])
                if arg_number == 3:
                    y_arr.append([x_pipe,y_pipe,z_pipe,x_scale,y_scale,z_scale])

        
        return X_arr, y_arr


def get_features_append(file_in):
    '''
    Extract the features data to be used for the model from all hits retrieved on the object
    C. Wibisono
    05/25 '26
    Parameter(s):
    file_in: fileinput pointer object 
    arg_number: (int) the target variables 1 for pipe only, 2 for scaling only, 3 for both
    Return(s):
    X: [arr] array of independent features variables
    y: [arr] array of dependent variables
    '''

    X_arr = []
    y_arr = []
    
    with open(file_in, mode='r') as fin:

        #Read header:
        fin.readline()
        
        data = {}

        
        #for i in range(100):
        while(1):
            line = fin.readline()
            if line == '':
                break
        
            else:                                
                row = line.split(',')
                obj_type = int(row[4].split('\n')[0])
                
                if row[0] in data.keys():
                    if obj_type == 1: 
                        x2_scint = float(row[1])
                        y2_scint = float(row[2])
                        z2_scint = float(row[3])
                        
                        data[row[0]][1].append(x2_scint)
                        data[row[0]][1].append(y2_scint)
                        data[row[0]][1].append(z2_scint)

                    if obj_type == 2: 
                        x3_scint = float(row[1])
                        y3_scint = float(row[2])
                        z3_scint = float(row[3])
                        
                        data[row[0]][2].append(x3_scint)
                        data[row[0]][2].append(y3_scint)
                        data[row[0]][2].append(z3_scint)


                    if obj_type == 3: 
                        x4_scint = float(row[1])
                        y4_scint = float(row[2])
                        z4_scint = float(row[3])
                        
                        data[row[0]][3].append(x4_scint)
                        data[row[0]][3].append(y4_scint)
                        data[row[0]][3].append(z4_scint)

                    if obj_type == 4: 
                        x_pipe = float(row[1])
                        y_pipe = float(row[2])
                        z_pipe = float(row[3])
                        
                        data[row[0]][4].append(x_pipe)
                        data[row[0]][4].append(y_pipe)
                        data[row[0]][4].append(z_pipe)
                
                    if obj_type == 5: 
                        x_scale = float(row[1])
                        y_scale = float(row[2])
                        z_scale = float(row[3])
                        
                        data[row[0]][5].append(x_scale)
                        data[row[0]][5].append(y_scale)
                        data[row[0]][5].append(z_scale)


                else:
                    if not data:
                        prev_id = row[0]
                        data[row[0]] =[[],[],[],[],[],[]]

                        if obj_type == 0:
                            x1_scint = float(row[1])
                            y1_scint = float(row[2])
                            z1_scint = float(row[3])
                
                            data[row[0]][0].append(x1_scint)
                            data[row[0]][0].append(y1_scint)
                            data[row[0]][0].append(z1_scint)

                    else:
                        print(data[prev_id][0],data[prev_id][1],data[prev_id][2],data[prev_id][3])
                        '''
                        Extract the features from the correlated file:
                        '''
                        dim_pipe = len(data[prev_id][4])
                        dim_scaling = len(data[prev_id][5])
                       
                        
                        if dim_pipe > 3 and dim_scaling > 3:
                            X_arr.append([
                                data[prev_id][0][0], data[prev_id][0][1], data[prev_id][0][2],
                                data[prev_id][1][0], data[prev_id][1][1], data[prev_id][1][2],
                                data[prev_id][2][0], data[prev_id][2][1], data[prev_id][2][2],
                                data[prev_id][3][0], data[prev_id][3][1], data[prev_id][3][2]
                                ])

                            y_arr.append([
                                data[prev_id][4][0], data[prev_id][4][1], data[prev_id][4][2],
                                data[prev_id][4][dim_pipe-3], data[prev_id][4][dim_pipe-2], data[prev_id][4][dim_pipe-1],
                                data[prev_id][5][0], data[prev_id][5][1], data[prev_id][5][2],
                                data[prev_id][5][dim_scaling-3], data[prev_id][5][dim_scaling-2], data[prev_id][5][dim_scaling-1]
                                ])
                         

                        del data
                        data = {}
                        data[row[0]] =[[],[],[],[],[],[]]
                        prev_id = row[0]
                        if obj_type == 0:
                            x1_scint = float(row[1])
                            y1_scint = float(row[2])
                            z1_scint = float(row[3])
                
                            data[row[0]][0].append(x1_scint)
                            data[row[0]][0].append(y1_scint)
                            data[row[0]][0].append(z1_scint)


        dim_pipe = len(data[prev_id][4])
        dim_scaling = len(data[prev_id][5])
        print(data[prev_id][0],data[prev_id][1],data[prev_id][2],data[prev_id][3])
        
        if dim_pipe > 3 and dim_scaling > 3:
            X_arr.append([
                data[prev_id][0][0], data[prev_id][0][1], data[prev_id][0][2],
                data[prev_id][1][0], data[prev_id][1][1], data[prev_id][1][2],
                data[prev_id][2][0], data[prev_id][2][1], data[prev_id][2][2],
                data[prev_id][3][0], data[prev_id][3][1], data[prev_id][3][2]
                ])

            y_arr.append([
                data[prev_id][4][0], data[prev_id][4][1], data[prev_id][4][2],
                data[prev_id][4][dim_pipe-3], data[prev_id][4][dim_pipe-2], data[prev_id][4][dim_pipe-1],
                data[prev_id][5][0], data[prev_id][5][1], data[prev_id][5][2],
                data[prev_id][5][dim_scaling-3], data[prev_id][5][dim_scaling-2], data[prev_id][5][dim_scaling-1]
                ])
                        
            
            del data
    
    
        
    return X_arr, y_arr



def write_header(fin, run_mode):
    '''
    Write header for model prediction result file
    C. Wibisono
    05/21 '26
    C. Wibisono
    Parameter(s):
    fin: file input pointer object where the prediction model is stored
    run_mode:(int) run_mode (1: first hit, 2: last hit, 3: all, 4: average)
    '''
    if (run_mode == 1 or run_mode == 2 or run_mode == 4):
        fin.write('#Pipe_Pos(x,y,z)'+','+'Scaling_Pos(x,y,z)'+'\n')





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
    dim_column = len(y_test[0]) #dimension of column 
    with open(fout, mode='w') as f:
        if dim_column == 3:
            for i in range(dim):
                f.write(str(y_test[i][0])+','+str(y_test[i][1])+','+str(y_test[i][2])+'\n')
        if dim_column == 6:
            for i in range(dim):
                f.write(str(y_test[i][0])+','+str(y_test[i][1])+','+str(y_test[i][2])+','+str(y_test[i][3])+','+str(y_test[i][4])+','+str(y_test[i][5])+'\n')

        if dim_column == 12:
            for i in range(dim):
                f.write(str(y_test[i][0])+','+str(y_test[i][1])+','+str(y_test[i][2])+','+ \
                        str(y_test[i][3])+','+str(y_test[i][4])+','+str(y_test[i][5])+','+ \
                        str(y_test[i][6])+','+str(y_test[i][7])+','+str(y_test[i][8])+','+ \
                        str(y_test[i][9])+','+str(y_test[i][10])+','+str(y_test[i][11])+'\n')


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
    dim_column = len(coords[0])
    if dim_column == 3:
        ax.scatter(coords[:,0],coords[:,1],coords[:,2],c='blue',marker='o',s=0.001)
    if dim_column == 6:
        ax.scatter(coords[:,0],coords[:,1],coords[:,2],c='blue',marker='o',s=0.001)
        ax.scatter(coords[:,3],coords[:,4],coords[:,5],c='red',marker='o',s=0.001)
    if dim_column == 12:
        ax.scatter(coords[:,0],coords[:,1],coords[:,2],c='blue',marker='o',s=0.001) #First Hit
        ax.scatter(coords[:,3],coords[:,4],coords[:,5],c='blue',marker='o',s=0.001) #Last Hit
        ax.scatter(coords[:,6],coords[:,7],coords[:,8],c='red',marker='o',s=0.001) #First Hit
        ax.scatter(coords[:,9],coords[:,10],coords[:,11],c='red',marker='o',s=0.001) #Last Hit

    plt.show()

def plot_prediction(fin):
    '''
    Plot the result from the model prediction
    C. Wibisono
    04/30 '26
    Parameter(s):
    fin: file input pointer object resulted from the write_features() function
    '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    with open(fin, mode='r') as f:
        prev = f.tell()
        line_0 = f.readline()
        row_0 = line_0.split(',')
        col_dim = len(row_0)
        print(col_dim)
        f.seek(prev)

        if col_dim == 3:
            arr = [[],[],[]]

            while(1):
                line = f.readline()
                if line == '':
                    break
                else:
                    row = line.split(',')
                    x = float(row[0])
                    y = float(row[1])
                    z = float(row[2])
                    arr[0].append(x)
                    arr[1].append(y)
                    arr[2].append(z)

            fig = plt.figure()
            ax = fig.add_subplot(projection = '3d')
            ax.scatter(arr[0],arr[1],arr[2],c='blue',marker='o',s=0.001)
            ax.view_init(elev = 0, azim = -90)

            del arr
            plt.show()

        if col_dim == 6:
            arr_1 = [[],[],[]]    
            arr_2 = [[],[],[]]    
            
            while(1):
                line = f.readline()
                if line == '':
                    break
                else:
                    row = line.split(',')
                    x_pipe = float(row[0])
                    y_pipe = float(row[1])
                    z_pipe = float(row[2])
                    arr_1[0].append(x_pipe)
                    arr_1[1].append(y_pipe)
                    arr_1[2].append(z_pipe)
                    
                    x_scaling = float(row[3])
                    y_scaling = float(row[4])
                    z_scaling = float(row[5])
                    arr_2[0].append(x_scaling)
                    arr_2[1].append(y_scaling)
                    arr_2[2].append(z_scaling)


            
            fig = plt.figure()
            ax = fig.add_subplot(projection = '3d')
            ax.scatter(arr_1[0],arr_1[1],arr_1[2],c='blue',marker='o',s=0.001)
            ax.scatter(arr_2[0],arr_2[1],arr_2[2],c='red',marker='o',s=0.001)
            ax.view_init(elev = 0, azim = -90)

            del arr_1
            del arr_2

            plt.show()


def plot_test_and_predict(f_test, f_predict):
    '''
    Plot the result from the model prediction and the true value from testing
    C. Wibisono
    04/30 '26
    Parameter(s):
    f_test: file input pointer object from the correlated muon hits on scintillators, pipe, and scaling
    f_predict: file input pointer object resulted from the write_features() function
    '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    with open(f_test, mode='r') as f1:
        arr_test_pipe=[[],[],[]]
        arr_test_scaling = [[],[],[]]

        #Read the header:
        f1.readline()

        while(1):
            line_test = f1.readline()
            if line_test == '':
                break
            else:
                row_test = line_test.split(',')
                x_pipe = float(row_test[13])
                y_pipe = float(row_test[14])
                z_pipe = float(row_test[15])
                arr_test_pipe[0].append(x_pipe)
                arr_test_pipe[1].append(y_pipe)
                arr_test_pipe[2].append(z_pipe)

                x_scaling = float(row_test[16])
                y_scaling = float(row_test[17])
                z_scaling = float(row_test[18])
                arr_test_scaling[0].append(x_scaling)
                arr_test_scaling[1].append(y_scaling)
                arr_test_scaling[2].append(z_scaling)



        with open(f_predict, mode='r') as f2:
            prev = f2.tell()
            line_0 = f2.readline()
            row_0 = line_0.split(',')
            col_dim = len(row_0)
            f2.seek(prev)

            if col_dim == 3:
                arr = [[],[],[]]

                while(1):
                    line = f2.readline()
                    if line == '':
                        break
                    else:
                        row = line.split(',')
                        x = float(row[0])
                        y = float(row[1])
                        z = float(row[2])
                        arr[0].append(x)
                        arr[1].append(y)
                        arr[2].append(z)

                fig = plt.figure()
                ax_predict = fig.add_subplot(1,2,1,projection = '3d')
                ax_test = fig.add_subplot(1,2,2, projection = '3d')

                ax_predict.scatter(arr[0],arr[1],arr[2],c='blue',marker='o',s=0.001)
                ax_test.scatter(arr_test_pipe[0],arr_test_pipe[1],arr_test_pipe[2],c='blue',marker='o',s=0.001)
                ax_predict.view_init(elev = 0, azim = -90)
                ax_test.view_init(elev = 0, azim = -90)

                del arr
                plt.show()

            if col_dim == 6:
                arr_1 = [[],[],[]]    
                arr_2 = [[],[],[]]    
            
                while(1):
                    line = f2.readline()
                    if line == '':
                        break
                    else:
                        row = line.split(',')
                        x_pipe = float(row[0])
                        y_pipe = float(row[1])
                        z_pipe = float(row[2])
                        arr_1[0].append(x_pipe)
                        arr_1[1].append(y_pipe)
                        arr_1[2].append(z_pipe)
                    
                        x_scaling = float(row[3])
                        y_scaling = float(row[4])
                        z_scaling = float(row[5])
                        arr_2[0].append(x_scaling)
                        arr_2[1].append(y_scaling)
                        arr_2[2].append(z_scaling)


            
                fig = plt.figure()
                ax_predict = fig.add_subplot(1,2,1,projection = '3d')
                ax_test = fig.add_subplot(1,2,2,projection = '3d')
                ax_predict.scatter(arr_1[0],arr_1[1],arr_1[2],c='blue',marker='o',s=0.001)
                ax_predict.scatter(arr_2[0],arr_2[1],arr_2[2],c='red',marker='o',s=0.001)
                ax_test.scatter(arr_test_pipe[0],arr_test_pipe[1],arr_test_pipe[2],c='blue',marker='o',s=0.001)
                ax_test.scatter(arr_test_scaling[0],arr_test_scaling[1],arr_test_scaling[2],c='red',marker='o',s=0.001)
                ax_predict.view_init(elev = 0, azim = -90)
                ax_predict.set_xlim(-200,200)
                ax_predict.set_ylim(-200,200)
                ax_predict.set_zlim(-200,200)
                ax_predict.set_title("Prediction from NN-model")
                ax_test.set_title("Truth value")
                ax_test.view_init(elev = 0, azim =-90)
                ax_test.set_xlim(-200,200)
                ax_test.set_ylim(-200,200)
                ax_test.set_zlim(-200,200)

                del arr_1
                del arr_2

                plt.show()

