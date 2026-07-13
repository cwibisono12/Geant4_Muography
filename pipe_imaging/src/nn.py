#!/usr/bin/env python3

import joblib
from sklearn.neural_network import MLPRegressor
import numpy as np

def get_parameters(X_train, y_train, layer, *, num_iter = 2000):
    '''
    Get the parameters of the Supervised NN based on Multi-Layer Perceptron
    C. Wibisono
    04/29 '26
    Parameter(s):
    X_train: [arr] independent features variables
    y_train: [arr] target
    layer: (tuple) number of neurons for each ith layer
    num_iter: (int) number of iterations (default = 2000)
    Return(s):
    coeff: [obj] parameters of the model
    '''

    coeff  = MLPRegressor(solver = 'lbfgs', alpha = 1e-5, 
            hidden_layer_sizes = layer, activation='relu', max_iter = num_iter)

    coeff.fit(X_train, y_train)

    return coeff

def initialize_model(X_train, y_train, layer, f_joblib):
    '''
    Initialize the model and store the model to joblib.
    C. Wibisono
    06/03 '26
    Parameter(s):
    X_train: [arr] independent features variables
    y_train: [arr] target
    layer: (tuple) number of neurons for each ith layer
    f_joblib: (obj) file pointer object to store the model
    '''

    init_model = MLPRegressor(solver = 'adam', alpha = 1e-1,
            hidden_layer_sizes = layer, activation='tanh', random_state = 12)

    init_model.partial_fit(X_train, y_train)

    joblib.dump(init_model, f_joblib)


def model_initialize(layer, f_joblib, *, activation_function = 'tanh'):
    '''
    Initialize the model and store the model to joblib.
    C. Wibisono
    06/03 '26
    Parameter(s):
    layer: (tuple) number of neurons for each ith layer
    f_joblib: (obj) file pointer object to store the model
    activation_function: (str) activation function eg. relu for linear, tanh for non-linear (default = tanh).
    '''

    init_model = MLPRegressor(solver = 'adam', alpha = 1e-1,
            hidden_layer_sizes = layer, activation = activation_function, random_state = 12)

    joblib.dump(init_model, f_joblib)


def retrain_model(X_train, y_train, f_joblib):
    '''
    Retrain the model for the new set of training data.
    C. Wibisono
    06/03 '26
    Parameter(s):
    X_train: [arr] independent features variables
    y_train: [arr] target
    f_joblib: (obj) file pointer object to store the model
    '''

    updated_model = joblib.load(f_joblib)

    updated_model.partial_fit(X_train, y_train)

    joblib.dump(updated_model, f_joblib)


def predict_outcome_from_file(X_test, f_joblib):
    '''
    Get the result of the model based on the retrained model
    C. Wibisono
    06/03 '26
    Parameter(s):
    X_test: [arr] independent features variables
    f_joblib: (obj) file pointer object where the model is stored
    Return(s):
    y_test: [arr] model prediction
    '''

    model = joblib.load(f_joblib)

    y_test = model.predict(X_test)

    return y_test


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


def get_model_score(X_test, y_test, model_param):
    '''
    Get the coefficient of determination on test data.
    C. Wibisono
    06/02 '26
    Parameter(s):
    X_test: [arr] independent features variables
    y_test: [arr] target variables
    model_param: [obj] param of model
    Return(s):
    score: (float) R2 of X_test w.r.t y_test
    '''

    score = model_param.score(X_test, y_test)

    return score

def get_model_score_from_file(X_test, y_test, f_joblib):
    '''
    Get the coefficient of determination on test data.
    C. Wibisono
    06/02 '26
    Parameter(s):
    X_test: [arr] independent features variables
    y_test: [arr] target variables
    f_joblib: [obj] file pointer object where the model is stored
    Return(s):
    score: (float) R2 of X_test w.r.t y_test
    '''

    model = joblib.load(f_joblib)

    score = model.score(X_test, y_test)

    return score

def store_score_result(fout, f_test, layer, num_epoch, score, f_model):
    '''
    Store the model score over the test data
    C. Wibisono
    06/02 '26
    Parameter(s):
    fout: file pointer object to store the model score result
    f_test: (str) file name of the test data assesed for the model score
    layer: (tuple) an array consisting the number of neurons for each layer
    num_epoch: (int) number of iterations (epoch)
    score: (float) R2 of X_test w.r.t y_test
    f_model: (str) file name of the model used
    '''
    from datetime import datetime

    dim = len(layer)
    temp = f_test.split('/')
    dim_temp = len(temp)
    test_fname = ''
    
    temp_model = f_model.split('/')
    dim_model = len(temp_model)
    model_fname = ''
    
    for i in range(dim_temp):
        if '.csv' in temp[i]:
            test_fname = test_fname + temp[i].split('.csv')[0]

    for j in range(dim_model):
        if '.joblib' in temp_model[j]:
            model_fname = model_fname + temp_model[j].split('.joblib')[0]
        if '.pth' in temp_model[j]:
            model_fname = model_fname + temp_model[j].split('.pth')[0]

    with open(fout, mode='a') as f:
        f.write('=========='+'\n')
        f.write('file_test: '+str(test_fname)+'\n')
        f.write('Timestamp: '+str(datetime.now())+'\n')
        f.write('Number of layer: '+','+str(dim)+'\n')
        f.write('file_model: '+str(model_fname)+'\n')

        for i in range(dim):
            f.write(str(layer[i])+'\n')

        f.write('Number of Iteration: '+','+str(num_epoch)+'\n')
        f.write('Model Score: '+','+str(score)+'\n')
        
def store_score_result_append(fout, f_test, layer, num_epoch, score_1, score_2, f_model_1, f_model_2):
    '''
    Store models score over the test data
    C. Wibisono
    06/02 '26
    Parameter(s):
    fout: file pointer object to store the model score result
    f_test: (str) file name of the test data assesed for the model score
    layer: (tuple) an array consisting the number of neurons for each layer
    num_epoch: (int) number of iterations (epoch)
    score: (float) R2 of X_test w.r.t y_test
    f_model: (str) file name of the model used
    '''
    from datetime import datetime

    dim = len(layer)
    temp = f_test.split('/')
    dim_temp = len(temp)
    test_fname = ''
    
    temp_model_1 = f_model_1.split('/')
    dim_model_1 = len(temp_model_1)
    model_fname_1 = ''
    
    temp_model_2 = f_model_2.split('/')
    dim_model_2 = len(temp_model_2)
    model_fname_2 = ''
    
    for i in range(dim_temp):
        if '.csv' in temp[i]:
            test_fname = test_fname + temp[i].split('.csv')[0]

    for j in range(dim_model_1):
        if '.joblib' in temp_model_1[j]:
            model_fname_1 = model_fname_1 + temp_model_1[j].split('.joblib')[0]
        if '.pth' in temp_model_1[j]:
            model_fname_1 = model_fname_1 + temp_model_1[j].split('.pth')[0]

    for k in range(dim_model_2):
        if '.joblib' in temp_model_2[k]:
            model_fname_2 = model_fname_2 + temp_model_2[k].split('.joblib')[0]
        if '.pth' in temp_model_2[k]:
            model_fname_2 = model_fname_2 + temp_model_2[k].split('.pth')[0]
    
    with open(fout, mode='a') as f:
        f.write('=========='+'\n')
        f.write('file_test: '+str(test_fname)+'\n')
        f.write('Timestamp: '+str(datetime.now())+'\n')
        f.write('Number of layer: '+','+str(dim)+'\n')
        f.write('file_model_1: '+str(model_fname_1)+'\n')
        f.write('file_model_2: '+str(model_fname_2)+'\n')

        for i in range(dim):
            f.write(str(layer[i])+'\n')

        f.write('Number of Iteration: '+','+str(num_epoch)+'\n')
        f.write('Model Score 1: '+','+str(score_1)+'\n')
        f.write('Model Score 2: '+','+str(score_2)+'\n')

def scatt_angle_transformer(data, key):
    '''
    Extract Scattering angle from the spatial coordinates features from the correlated file
    C. Wibisono
    06/23 '26
    Parameter(s):
    data: (dict) dictionary containing spatial information of the hits in the correlated file.
    key: event_ID
    Return(s):
    scatt_angle: (deg) scattering angle.
    '''

    p1 = np.array([data[key][0][0], data[key][0][1], data[key][0][2]])
    p1_b = np.array([data[key][1][0], data[key][1][1], data[key][1][2]])
    p2 = np.array([data[key][2][0], data[key][2][1], data[key][2][2]])
    p2_b = np.array([data[key][3][0], data[key][3][1], data[key][3][2]])
    
    #Generate Direction vector:
    #Incoming vector:
    num_v1 = p1_b - p1
    denum_v1 = np.linalg.norm(num_v1)
    v1 = num_v1/denum_v1

    #Outgoing vector:
    num_v2 = p2_b - p2
    denum_v2 = np.linalg.norm(num_v2)
    v2 = num_v2/denum_v2

    #Generate scattering angle:
    theta_deg = np.rad2deg(np.arccos(np.dot(v1,v2)))

    return theta_deg

def cart_to_cylind(x, y, z):
    '''
    Transform the cartesian coordinates to cylindrical coordinates to better match target geometry.
    C. Wibisono
    06/29 '26
    Parameter(s):
    x: (float) x coordinate
    y: (float) y coordinate
    z: (float) z coordinate
    Return(s):
    r: (float) radius coordinate
    x_hat: (float) unit vector of angular coordinates projected onto the x-axis
    z_hat: (float) unit vector of angular coordinates projected onto the z-axis
    y: (float) axial symmetry
    '''

    r = ((z**2.) + (x**2.))**0.5
    
    theta = np.arctan2(z, x)
    
    #Transform theta into pairs of cos \theta and sin \theta
    x_hat = np.cos(theta)
    z_hat = np.sin(theta)

    return r, x_hat, z_hat, y

def cart_to_cylind_theta(x, y, z):
    '''
    Transform the cartesian coordinates to cylindrical coordinates to better match target geometry.
    C. Wibisono
    07/03 '26
    Parameter(s):
    x: (float) x coordinate
    y: (float) y coordinate
    z: (float) z coordinate
    Return(s):
    r: (float) radius coordinate
    theta: (rad) polar angle
    y: (float) axial symmetry
    '''

    r = ((z**2.) + (x**2.))**0.5
    
    theta = np.arctan2(z, x)
    
    return r, theta, y

def cylind_to_cart(r, x_hat, z_hat,  y):
    '''
    Transform the cylindrical coordinates to caretesian coordinates.
    C. Wibisono
    06/29 '26
    Parameter(s):
    r: (float) radius coordinate
    x_hat:(float) unit vector of angular coordinates projected onto the x-axis
    z_hat: (float) unit vector of angular coordinates projected onto the z-axis
    y: (float) axial symmetry
    Return(s):
    x: (float) x coordinate.
    y: (float) y coordinate.
    z: (float) z coordinate.
    '''

    #Infer the angular coordinate from the cartesian unit vector:
    theta = np.arctan2(z_hat, x_hat)

    #Transform the cylindrical to cartesian coordinate:
    x = r*(np.cos(theta))
    y = y
    z = r*(np.sin(theta))
    
    return x, y, z

def cylind_to_cart_theta(r, theta,  y):
    '''
    Transform the cylindrical coordinates to caretesian coordinates.
    C. Wibisono
    06/29 '26
    Parameter(s):
    r: (float) radius coordinate
    thata: (rad) polar angle
    y: (float) axial symmetry
    Return(s):
    x: (float) x coordinate.
    y: (float) y coordinate.
    z: (float) z coordinate.
    '''

    #Transform the cylindrical to cartesian coordinate:
    x = r*(np.cos(theta))
    y = y
    z = r*(np.sin(theta))
    
    return x, y, z

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


def get_features_append(file_in, *, arg_number = 2, theta_scatt = 0.5):
    '''
    Extract the features data to be used for the model from all hits retrieved on the object
    C. Wibisono
    05/25 '26
    Parameter(s):
    file_in: fileinput pointer object 
    arg_number: (int) number of target variables (2:) for first and last hits , (3:) for the first, last and the other hits
    theta_scatt: (deg) allowed scattering angle for event selection.
    (4:), the first and last hits and the last hit for incoming muon and the first hit for outgoing muon.
    (5:), the first and last hits and with addition of a new feature as interaction term.
    (6:), the first, last and other hits with addition of a new feature as interaction term.
    (7:), the first, last and other hits with addition of a new feature as intearaction term. (Target coordinates change to cylindrical).
    (8:), the first, last and other hits with addition of a new feature as intearaction term. (Target coordinates change to cylindrical with theta as polar angle).
    theta_scatt: (float) scattering angle between the incoming and outgoing muon from the scintillator hits for event selection.
    (9:), the first, last and other hits with addition of a new feature as intearaction term. (Target coordinates change to cylindrical with rescaled r and polar coo    rdinates).
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
                        scatt_angle = scatt_angle_transformer(data, prev_id)
                        print(data[prev_id][0],data[prev_id][1],data[prev_id][2],data[prev_id][3], scatt_angle)
                        '''
                        Extract the features from the correlated file:
                        '''
                        dim_pipe = len(data[prev_id][4])
                        mid_dim_pipe = 3*int(((dim_pipe/3)//2))
                        dim_scaling = len(data[prev_id][5])
                        mid_dim_scaling = 3*int(((dim_scaling/3)//2))
                        
                        if dim_pipe > 6 and dim_scaling > 6:

                            if arg_number == 2:
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

                            if arg_number == 3:
                                X_arr.append([
                                    data[prev_id][0][0], data[prev_id][0][1], data[prev_id][0][2],
                                    data[prev_id][1][0], data[prev_id][1][1], data[prev_id][1][2],
                                    data[prev_id][2][0], data[prev_id][2][1], data[prev_id][2][2],
                                    data[prev_id][3][0], data[prev_id][3][1], data[prev_id][3][2]
                                    ])
                                y_arr.append([
                                    data[prev_id][4][0], data[prev_id][4][1], data[prev_id][4][2],
                                    data[prev_id][4][mid_dim_pipe], data[prev_id][4][mid_dim_pipe+1], data[prev_id][4][mid_dim_pipe+2],
                                    data[prev_id][4][dim_pipe-3], data[prev_id][4][dim_pipe-2], data[prev_id][4][dim_pipe-1],
                                    data[prev_id][5][0], data[prev_id][5][1], data[prev_id][5][2],
                                    data[prev_id][5][mid_dim_scaling], data[prev_id][5][mid_dim_scaling+1], data[prev_id][5][mid_dim_scaling+2],
                                    data[prev_id][5][dim_scaling-3], data[prev_id][5][dim_scaling-2], data[prev_id][5][dim_scaling-1]
                                    ])

                            if arg_number == 4:
                                z_pos_in = []
                                z_pos_out = []
                                num_iteration = dim_scaling // 3
                                flag_valid = 0

                                for i in range(num_iteration):
                                    z_ind = 3*i + 2
                                    z_pos = data[prev_id][5][z_ind]
                                    if z_pos > 0:
                                        z_pos_in.append(z_ind)
                                    if z_pos < 0:
                                        z_pos_out.append(z_ind)

                                #Get the indices for the last incoming muon and the first outgoing muon:
                                if len(z_pos_in) > 0 and len(z_pos_out) > 0:
                                    last_in_index = z_pos_in[len(z_pos_in)-1]
                                    first_out_index = z_pos_out[0]
                                    flag_valid = 1

                                del z_pos_in
                                del z_pos_out
                                
                                if flag_valid == 1: #Only retrieve the event that has both incoming and outgoing muons over the scaling.
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
                                        data[prev_id][5][last_in_index-2], data[prev_id][5][last_in_index-1], data[prev_id][5][last_in_index],
                                        data[prev_id][5][first_out_index-2], data[prev_id][5][first_out_index-1], data[prev_id][5][first_out_index],
                                        data[prev_id][5][dim_scaling-3], data[prev_id][5][dim_scaling-2], data[prev_id][5][dim_scaling-1]
                                        ])


                            if arg_number == 5:
                                X_arr.append([
                                    data[prev_id][0][0], data[prev_id][0][1], data[prev_id][0][2],
                                    data[prev_id][1][0], data[prev_id][1][1], data[prev_id][1][2],
                                    data[prev_id][2][0], data[prev_id][2][1], data[prev_id][2][2],
                                    data[prev_id][3][0], data[prev_id][3][1], data[prev_id][3][2],
                                    scatt_angle
                                    ])

                                y_arr.append([
                                    data[prev_id][4][0], data[prev_id][4][1], data[prev_id][4][2],
                                    data[prev_id][4][dim_pipe-3], data[prev_id][4][dim_pipe-2], data[prev_id][4][dim_pipe-1],
                                    data[prev_id][5][0], data[prev_id][5][1], data[prev_id][5][2],
                                    data[prev_id][5][dim_scaling-3], data[prev_id][5][dim_scaling-2], data[prev_id][5][dim_scaling-1]
                                    ])


                            if arg_number == 6:
                                #Require only event in which the scattering angle is greater than 0.5 deg
                                if scatt_angle >= theta_scatt:
                                    X_arr.append([
                                        data[prev_id][0][0], data[prev_id][0][1], data[prev_id][0][2],
                                        data[prev_id][1][0], data[prev_id][1][1], data[prev_id][1][2],
                                        data[prev_id][2][0], data[prev_id][2][1], data[prev_id][2][2],
                                        data[prev_id][3][0], data[prev_id][3][1], data[prev_id][3][2],
                                        scatt_angle
                                        ])
                                    y_arr.append([
                                        data[prev_id][4][0], data[prev_id][4][1], data[prev_id][4][2],
                                        data[prev_id][4][mid_dim_pipe], data[prev_id][4][mid_dim_pipe+1], data[prev_id][4][mid_dim_pipe+2],
                                        data[prev_id][4][dim_pipe-3], data[prev_id][4][dim_pipe-2], data[prev_id][4][dim_pipe-1],
                                        data[prev_id][5][0], data[prev_id][5][1], data[prev_id][5][2],
                                        data[prev_id][5][mid_dim_scaling], data[prev_id][5][mid_dim_scaling+1], data[prev_id][5][mid_dim_scaling+2],
                                        data[prev_id][5][dim_scaling-3], data[prev_id][5][dim_scaling-2], data[prev_id][5][dim_scaling-1]
                                        ])

                            if arg_number == 7:
                                #Require only event in which the scattering angle is greater than 0.5 deg
                                if scatt_angle >= theta_scatt:
                                    X_arr.append([
                                        data[prev_id][0][0], data[prev_id][0][1], data[prev_id][0][2],
                                        data[prev_id][1][0], data[prev_id][1][1], data[prev_id][1][2],
                                        data[prev_id][2][0], data[prev_id][2][1], data[prev_id][2][2],
                                        data[prev_id][3][0], data[prev_id][3][1], data[prev_id][3][2],
                                        scatt_angle
                                        ])
                                   
                                    #Transform to cylindrical coordinates:
                                    r0, x0, z0, y0 = cart_to_cylind(data[prev_id][4][0], data[prev_id][4][1], data[prev_id][4][2])
                                    r1, x1, z1, y1 = cart_to_cylind(data[prev_id][4][mid_dim_pipe], data[prev_id][4][mid_dim_pipe+1], data[prev_id][4][mid_dim_pipe+2])
                                    r2, x2, z2, y2 = cart_to_cylind(data[prev_id][4][dim_pipe-3], data[prev_id][4][dim_pipe-2], data[prev_id][4][dim_pipe-1])
                                    r3, x3, z3, y3 = cart_to_cylind(data[prev_id][5][0], data[prev_id][5][1], data[prev_id][5][2])
                                    r4, x4, z4, y4 = cart_to_cylind(data[prev_id][5][mid_dim_scaling], data[prev_id][5][mid_dim_scaling+1], data[prev_id][5][mid_dim_scaling+2])
                                    r5, x5, z5, y5 = cart_to_cylind(data[prev_id][5][dim_scaling-3], data[prev_id][5][dim_scaling-2], data[prev_id][5][dim_scaling-1])
                                    
                                    y_arr.append([
                                        r0, x0, z0, y0,
                                        r1, x1, z1, y1,
                                        r2, x2, z2, y2,
                                        r3, x3, z3, y3,
                                        r4, x4, z4, y4,
                                        r5, x5, z5, y5
                                        ])

                            if arg_number == 8:
                                #Require only event in which the scattering angle is greater than 0.5 deg
                                if scatt_angle >= theta_scatt:
                                    X_arr.append([
                                        data[prev_id][0][0], data[prev_id][0][1], data[prev_id][0][2],
                                        data[prev_id][1][0], data[prev_id][1][1], data[prev_id][1][2],
                                        data[prev_id][2][0], data[prev_id][2][1], data[prev_id][2][2],
                                        data[prev_id][3][0], data[prev_id][3][1], data[prev_id][3][2],
                                        scatt_angle
                                        ])
                                   
                                    #Transform to cylindrical coordinates:
                                    r0, theta0, y0 = cart_to_cylind_theta(data[prev_id][4][0], data[prev_id][4][1], data[prev_id][4][2])
                                    r1, theta1, y1 = cart_to_cylind_theta(data[prev_id][4][mid_dim_pipe], data[prev_id][4][mid_dim_pipe+1], data[prev_id][4][mid_dim_pipe+2])
                                    r2, theta2, y2 = cart_to_cylind_theta(data[prev_id][4][dim_pipe-3], data[prev_id][4][dim_pipe-2], data[prev_id][4][dim_pipe-1])
                                    r3, theta3, y3 = cart_to_cylind_theta(data[prev_id][5][0], data[prev_id][5][1], data[prev_id][5][2])
                                    r4, theta4, y4 = cart_to_cylind_theta(data[prev_id][5][mid_dim_scaling], data[prev_id][5][mid_dim_scaling+1], data[prev_id][5][mid_dim_scaling+2])
                                    r5, theta5, y5 = cart_to_cylind_theta(data[prev_id][5][dim_scaling-3], data[prev_id][5][dim_scaling-2], data[prev_id][5][dim_scaling-1])
                                    
                                    y_arr.append([
                                        r0, theta0, y0,
                                        r1, theta1, y1,
                                        r2, theta2, y2,
                                        r3, theta3, y3,
                                        r4, theta4, y4,
                                        r5, theta5, y5
                                        ])

                            if arg_number == 9:
                                #Require only event in which the scattering angle is greater than 0.5 deg
                                if scatt_angle >= theta_scatt:
                                    X_arr.append([
                                        data[prev_id][0][0], data[prev_id][0][1], data[prev_id][0][2],
                                        data[prev_id][1][0], data[prev_id][1][1], data[prev_id][1][2],
                                        data[prev_id][2][0], data[prev_id][2][1], data[prev_id][2][2],
                                        data[prev_id][3][0], data[prev_id][3][1], data[prev_id][3][2],
                                        scatt_angle
                                        ])
                                   
                                    #Transform to cylindrical coordinates:
                                    r0, x0, z0, y0 = cart_to_cylind(data[prev_id][4][0], data[prev_id][4][1], data[prev_id][4][2])
                                    r1, x1, z1, y1 = cart_to_cylind(data[prev_id][4][mid_dim_pipe], data[prev_id][4][mid_dim_pipe+1], data[prev_id][4][mid_dim_pipe+2])
                                    r2, x2, z2, y2 = cart_to_cylind(data[prev_id][4][dim_pipe-3], data[prev_id][4][dim_pipe-2], data[prev_id][4][dim_pipe-1])
                                    r3, x3, z3, y3 = cart_to_cylind(data[prev_id][5][0], data[prev_id][5][1], data[prev_id][5][2])
                                    r4, x4, z4, y4 = cart_to_cylind(data[prev_id][5][mid_dim_scaling], data[prev_id][5][mid_dim_scaling+1], data[prev_id][5][mid_dim_scaling+2])
                                    r5, x5, z5, y5 = cart_to_cylind(data[prev_id][5][dim_scaling-3], data[prev_id][5][dim_scaling-2], data[prev_id][5][dim_scaling-1])
                                    
                                    #Rescale the r, cos \theta, and sin \theta coordinates by multiplying them with radius.
                                    y_arr.append([
                                        r0*r0, r0*x0, r0*z0, y0,
                                        r1*r1, r1*x1, r1*z1, y1,
                                        r2*r2, r2*x2, r2*z2, y2,
                                        r3*r3, r3*x3, r3*z3, y3,
                                        r4*r4, r4*x4, r4*z4, y4,
                                        r5*r5, r5*x5, r5*z5, y5
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
        mid_dim_pipe = 3*int(((dim_pipe/3)//2))
        dim_scaling = len(data[prev_id][5])
        mid_dim_scaling = 3*int(((dim_scaling/3)//2))

        scatt_angle = scatt_angle_transformer(data, prev_id)
        print(data[prev_id][0],data[prev_id][1],data[prev_id][2],data[prev_id][3], scatt_angle)
        
        if dim_pipe > 6 and dim_scaling > 6:

            if arg_number == 2:
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

            if arg_number == 3:
                X_arr.append([
                    data[prev_id][0][0], data[prev_id][0][1], data[prev_id][0][2],
                    data[prev_id][1][0], data[prev_id][1][1], data[prev_id][1][2],
                    data[prev_id][2][0], data[prev_id][2][1], data[prev_id][2][2],
                    data[prev_id][3][0], data[prev_id][3][1], data[prev_id][3][2]
                    ])
                y_arr.append([
                    data[prev_id][4][0], data[prev_id][4][1], data[prev_id][4][2],
                    data[prev_id][4][mid_dim_pipe], data[prev_id][4][mid_dim_pipe+1], data[prev_id][4][mid_dim_pipe+2],
                    data[prev_id][4][dim_pipe-3], data[prev_id][4][dim_pipe-2], data[prev_id][4][dim_pipe-1],
                    data[prev_id][5][0], data[prev_id][5][1], data[prev_id][5][2],
                    data[prev_id][5][mid_dim_scaling], data[prev_id][5][mid_dim_scaling+1], data[prev_id][5][mid_dim_scaling+2],
                    data[prev_id][5][dim_scaling-3], data[prev_id][5][dim_scaling-2], data[prev_id][5][dim_scaling-1]
                    ])
                        
            if arg_number == 4:
                z_pos_in = []
                z_pos_out = []
                num_iteration = dim_scaling // 3
                flag_valid = 0

                for i in range(num_iteration):
                    z_ind = 3*i + 2
                    z_pos = data[prev_id][5][z_ind]
                    if z_pos > 0:
                        z_pos_in.append(z_ind)
                    if z_pos < 0:
                        z_pos_out.append(z_ind)

                if len(z_pos_in) > 0 and len(z_pos_out) > 0:
                    #Get the indices for the last incoming muon and the first outgoing muon:
                    last_in_index = z_pos_in[len(z_pos_in)-1]
                    first_out_index = z_pos_out[0]
                    flag_valid = 1

                del z_pos_in
                del z_pos_out
                if flag_valid == 1:
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
                        data[prev_id][5][last_in_index-2], data[prev_id][5][last_in_index-1], data[prev_id][5][last_in_index],
                        data[prev_id][5][first_out_index-2], data[prev_id][5][first_out_index-1], data[prev_id][5][first_out_index],
                        data[prev_id][5][dim_scaling-3], data[prev_id][5][dim_scaling-2], data[prev_id][5][dim_scaling-1]
                        ])


            if arg_number == 5:             
                X_arr.append([
                    data[prev_id][0][0], data[prev_id][0][1], data[prev_id][0][2],
                    data[prev_id][1][0], data[prev_id][1][1], data[prev_id][1][2],
                    data[prev_id][2][0], data[prev_id][2][1], data[prev_id][2][2],
                    data[prev_id][3][0], data[prev_id][3][1], data[prev_id][3][2],
                    scatt_angle
                    ])

                y_arr.append([
                    data[prev_id][4][0], data[prev_id][4][1], data[prev_id][4][2],
                    data[prev_id][4][dim_pipe-3], data[prev_id][4][dim_pipe-2], data[prev_id][4][dim_pipe-1],
                    data[prev_id][5][0], data[prev_id][5][1], data[prev_id][5][2],
                    data[prev_id][5][dim_scaling-3], data[prev_id][5][dim_scaling-2], data[prev_id][5][dim_scaling-1]
                    ])

            if arg_number == 6:
                #Require only event in which the scattering angle is greather than 0.5 deg
                if scatt_angle > theta_scatt:
                    X_arr.append([
                        data[prev_id][0][0], data[prev_id][0][1], data[prev_id][0][2],
                        data[prev_id][1][0], data[prev_id][1][1], data[prev_id][1][2],
                        data[prev_id][2][0], data[prev_id][2][1], data[prev_id][2][2],
                        data[prev_id][3][0], data[prev_id][3][1], data[prev_id][3][2],
                        scatt_angle
                        ])
                    y_arr.append([
                        data[prev_id][4][0], data[prev_id][4][1], data[prev_id][4][2],
                        data[prev_id][4][mid_dim_pipe], data[prev_id][4][mid_dim_pipe+1], data[prev_id][4][mid_dim_pipe+2],
                        data[prev_id][4][dim_pipe-3], data[prev_id][4][dim_pipe-2], data[prev_id][4][dim_pipe-1],
                        data[prev_id][5][0], data[prev_id][5][1], data[prev_id][5][2],
                        data[prev_id][5][mid_dim_scaling], data[prev_id][5][mid_dim_scaling+1], data[prev_id][5][mid_dim_scaling+2],
                        data[prev_id][5][dim_scaling-3], data[prev_id][5][dim_scaling-2], data[prev_id][5][dim_scaling-1]
                        ])

            if arg_number == 7:
                #Require only event in which the scattering angle is greather than 0.5 deg
                if scatt_angle > theta_scatt:
                    X_arr.append([
                        data[prev_id][0][0], data[prev_id][0][1], data[prev_id][0][2],
                        data[prev_id][1][0], data[prev_id][1][1], data[prev_id][1][2],
                        data[prev_id][2][0], data[prev_id][2][1], data[prev_id][2][2],
                        data[prev_id][3][0], data[prev_id][3][1], data[prev_id][3][2],
                        scatt_angle
                        ])
                    
                    r0, x0, z0, y0 = cart_to_cylind(data[prev_id][4][0], data[prev_id][4][1], data[prev_id][4][2])
                    r1, x1, z1, y1 = cart_to_cylind(data[prev_id][4][mid_dim_pipe], data[prev_id][4][mid_dim_pipe+1], data[prev_id][4][mid_dim_pipe+2])
                    r2, x2, z2, y2 = cart_to_cylind(data[prev_id][4][dim_pipe-3], data[prev_id][4][dim_pipe-2], data[prev_id][4][dim_pipe-1])
                    r3, x3, z3, y3 = cart_to_cylind(data[prev_id][5][0], data[prev_id][5][1], data[prev_id][5][2])
                    r4, x4, z4, y4 = cart_to_cylind(data[prev_id][5][mid_dim_scaling], data[prev_id][5][mid_dim_scaling+1], data[prev_id][5][mid_dim_scaling+2])
                    r5, x5, z5, y5 = cart_to_cylind(data[prev_id][5][dim_scaling-3], data[prev_id][5][dim_scaling-2], data[prev_id][5][dim_scaling-1])
                    
                    y_arr.append([
                        r0, x0, z0, y0,
                        r1, x1, z1, y1,
                        r2, x2, z2, y2,
                        r3, x3, z3, y3,
                        r4, x4, z4, y4,
                        r5, x5, z5, y5
                        ])


            if arg_number == 8:
                #Require only event in which the scattering angle is greather than 0.5 deg
                if scatt_angle > theta_scatt:
                    X_arr.append([
                        data[prev_id][0][0], data[prev_id][0][1], data[prev_id][0][2],
                        data[prev_id][1][0], data[prev_id][1][1], data[prev_id][1][2],
                        data[prev_id][2][0], data[prev_id][2][1], data[prev_id][2][2],
                        data[prev_id][3][0], data[prev_id][3][1], data[prev_id][3][2],
                        scatt_angle
                        ])
                    
                    r0, theta0, y0 = cart_to_cylind_theta(data[prev_id][4][0], data[prev_id][4][1], data[prev_id][4][2])
                    r1, theta1, y1 = cart_to_cylind_theta(data[prev_id][4][mid_dim_pipe], data[prev_id][4][mid_dim_pipe+1], data[prev_id][4][mid_dim_pipe+2])
                    r2, theta2, y2 = cart_to_cylind_theta(data[prev_id][4][dim_pipe-3], data[prev_id][4][dim_pipe-2], data[prev_id][4][dim_pipe-1])
                    r3, theta3, y3 = cart_to_cylind_theta(data[prev_id][5][0], data[prev_id][5][1], data[prev_id][5][2])
                    r4, theta4, y4 = cart_to_cylind_theta(data[prev_id][5][mid_dim_scaling], data[prev_id][5][mid_dim_scaling+1], data[prev_id][5][mid_dim_scaling+2])
                    r5, theta5, y5 = cart_to_cylind_theta(data[prev_id][5][dim_scaling-3], data[prev_id][5][dim_scaling-2], data[prev_id][5][dim_scaling-1])
                    
                    y_arr.append([
                        r0, theta0, y0,
                        r1, theta1, y1,
                        r2, theta2, y2,
                        r3, theta3, y3,
                        r4, theta4, y4,
                        r5, theta5, y5
                        ])

            if arg_number == 9:
                #Require only event in which the scattering angle is greather than 0.5 deg
                if scatt_angle > theta_scatt:
                    X_arr.append([
                        data[prev_id][0][0], data[prev_id][0][1], data[prev_id][0][2],
                        data[prev_id][1][0], data[prev_id][1][1], data[prev_id][1][2],
                        data[prev_id][2][0], data[prev_id][2][1], data[prev_id][2][2],
                        data[prev_id][3][0], data[prev_id][3][1], data[prev_id][3][2],
                        scatt_angle
                        ])
                    
                    r0, x0, z0, y0 = cart_to_cylind(data[prev_id][4][0], data[prev_id][4][1], data[prev_id][4][2])
                    r1, x1, z1, y1 = cart_to_cylind(data[prev_id][4][mid_dim_pipe], data[prev_id][4][mid_dim_pipe+1], data[prev_id][4][mid_dim_pipe+2])
                    r2, x2, z2, y2 = cart_to_cylind(data[prev_id][4][dim_pipe-3], data[prev_id][4][dim_pipe-2], data[prev_id][4][dim_pipe-1])
                    r3, x3, z3, y3 = cart_to_cylind(data[prev_id][5][0], data[prev_id][5][1], data[prev_id][5][2])
                    r4, x4, z4, y4 = cart_to_cylind(data[prev_id][5][mid_dim_scaling], data[prev_id][5][mid_dim_scaling+1], data[prev_id][5][mid_dim_scaling+2])
                    r5, x5, z5, y5 = cart_to_cylind(data[prev_id][5][dim_scaling-3], data[prev_id][5][dim_scaling-2], data[prev_id][5][dim_scaling-1])
                    
                    y_arr.append([
                        r0*r0, r0*x0, r0*z0, y0,
                        r1*r1, r1*x1, r1*z1, y1,
                        r2*r2, r2*x2, r2*z2, y2,
                        r3*r3, r3*x3, r3*z3, y3,
                        r4*r4, r4*x4, r4*z4, y4,
                        r5*r5, r5*x5, r5*z5, y5
                        ])

            del data
        
    return X_arr, y_arr

def select_features(X_arr, *, threshold = 0.16):
    '''
    Reduce dimension of higher dimensional data that does not meet the threshold
    of variance.
    C. Wibisono
    06/09 '26
    Parameter(s):
    X_arr: [arr] independent features
    threshold: (float) threshold value to reduce data dimension
    Return(s):
    X_arr_new: [arr] reduced independent features
    '''

    from sklearn.feature_selection import VarianceThreshold
    
    sel = VarianceThreshold(threshold)
    
    X_arr_new = sel.fit_transform(X_arr)
    
    return X_arr_new

def preprocess_selected_features(X_arr):
    '''
    Transform each selected feature by transforming them relative to the center of the training data.
    C. Wibisono
    06/11 '26
    Parameter(s):
    X_arr: [arr] independent features
    Return(s):
    X_arr_scaled: [arr] transformed features after preprocessing.
    '''

    from sklearn import preprocessing
    
    scaler = preprocessing.StandardScaler().fit(X_arr)

    X_arr_scaled = scaler.transform(X_arr)

    return X_arr_scaled

def initialize_preprocess_batch():
    '''
    Initialize the batch transformer
    '''

    from sklearn import preprocessing

    scaler = preprocessing.StandardScaler()

    return scaler

def preprocess_selected_features_batch(transformer,X_arr):
    '''
    Transform each selected feature by transforming them relative to the center of the training data
    that come in per batch.
    C. Wibisono
    06/11 '26
    Parameter(s):
    transformer: [obj] initialized standardscaler transformer object
    X_arr: [arr] independent features
    Return(s):
    X_arr_scaled: [arr] transformed features after preprocessing.
    '''

    from sklearn import preprocessing
    
    transformer.partial_fit(X_arr)

    X_arr_scaled = scaler.transform(X_arr)

    return X_arr_scaled

def scaler_initialize(f_transform, *, with_mean_included = True):
    '''
    Initialize an empty scaler to be used to scale the data over the data batches 
    and memory keeping to retrieve the global mean and variances over all training data.
    C. Wibisono
    06/15 '26
    Parameter(s):
    f_transform: (obj) file pointer object to store the transformer.
    with_mean_included: (bool) center the data before scaling. (Default = True)
    '''

    from sklearn.preprocessing import StandardScaler

    scaler_init = StandardScaler(with_mean = with_mean_included)

    joblib.dump(scaler_init, f_transform)

def scaler_update(X_train, f_transform):
    '''
    Update the transformer file to refit the features when the new data comes in.
    C. Wibisono
    06/15 '26
    Parameter(s):
    X_train: [arr] independent features to be rescaled.
    f_transform: (obj) file pointer object to update the transformer.
    '''

    #Load the transformer:
    updated_scaler = joblib.load(f_transform)

    #Update the global mean and variance with the new data:
    updated_scaler.partial_fit(X_train)

    #Update the transformer file:
    joblib.dump(updated_scaler, f_transform)


def rescale_features(X_train, f_transform):
    '''
    Rescale the features given the transformer file.
    C. Wibisono
    06/15 '26:
    Parameter(s):
    X_train: [arr] independent features to be rescaled
    f_transform: (obj) file pointer object of the transformer to rescale the features.
    Return(s):
    X_train_scaled: [arr] rescaled independent features.
    '''

    #Load the transformer file:
    scaler = joblib.load(f_transform)

    #Rescale the features:
    X_train_scaled = scaler.transform(X_train)

    return X_train_scaled

def inverse_transform(y_train_rescaled, f_transform):
    '''
    Perform the inverse rescaling transformation given the transformer file.
    C. Wibisono
    06/18 '26:
    Parameter(s):
    y_train_rescaled: [arr] rescaled target(s).
    f_transform: (obj) file pointer object of the transformer to rescale the features.
    Return(s):
    y_train_inv: [arr] inverse rescale target(s).
    '''

    #Load the transformer file:
    scaler = joblib.load(f_transform)

    #Inverse transform the target:
    y_train_inv = scaler.inverse_transform(y_train_rescaled)

    return y_train_inv

def summary_statistics(X_train, f_train, f_out):
    '''
    Get the statistics summary for features.
    C. Wibisono
    06/23 '26
    Parameter(s):
    X_train: [arr] independent features.
    f_train: location of training files.
    f_out: file to store the summary statistics
    '''
    from datetime import datetime
    
    temp = f_train.split('/')
    dim_temp = len(temp)
    train_fname = ''

    for i in range(dim_temp):
        if '.csv' in temp[i]:
            train_fname = train_fname + temp[i].split('.csv')[0]
    

    #Calculate statistics summary for each features in X_train:
    dim_column = len(X_train[0])

    X_train_np = np.array(X_train)

    with open(f_out, mode='a') as f:
        f.write('=========='+'\n')
        f.write('file_training: '+str(train_fname)+'\n')
        f.write('Timestamp: '+str(datetime.now())+'\n')
        for j in range(dim_column):
            f.write('mean of X_'+str(j)+' :'+','+str(np.mean(X_train_np[:,j]))+'\n')
            f.write('SD of X_'+str(j)+' :'+','+str(np.var(X_train_np[:,j]))+'\n')

def transform_feature_from_array(X_train, *, arg_mode = 5):
    '''
    Transform a feature from a given array.
    C. Wibisono
    06/24 '26
    Parameter(s):
    X_train: [arr] independent features.
    arg_mode: (int) number of target variables (see get_features_append function).
    '''

    dim_row = len(X_train)
    dim_column = len(X_train[0])
    
    #Get the mean of the last feature if arg_mode is listed below.
    if arg_mode == 5 or arg_mode == 6 or arg_mode == 7 or arg_mode == 8 or arg_mode == 9:
        X_train_np = np.array(X_train)
        mean_last_Xcol = round(np.mean(X_train_np[:,dim_column - 1]),2)
        
        del X_train_np

        #Overwrite the last column:
        for i in range(dim_row):
            X_train[i][dim_column -1] = mean_last_Xcol

        
    return X_train


def split_target(y_train, *, arg_mode = 7):
    '''
    Split the target variables into two parts.
    C. Wibisono
    07/01 '26
    Parameter(s):
    y_train: [arr] target variables
    arg_mode: (int) mode  of target variables (see get_features_append function).
    Return(s):
    y_train_1: [arr] splitted 1st targets
    y_train_2: [arr] splitted 2nd targets
    '''

    dim_row = len(y_train)
    
    y_train_1 = []
    y_train_2 = []
    dim_row = len(y_train)
    if arg_mode == 7 or arg_mode == 9:
        for i in range(dim_row):
            y_train_1.append([
                y_train[i][0], y_train[i][3], y_train[i][4], y_train[i][7],
                y_train[i][8], y_train[i][11], y_train[i][12], y_train[i][15],
                y_train[i][16], y_train[i][19], y_train[i][20], y_train[i][23]
                ])
            y_train_2.append([
                y_train[i][1], y_train[i][2], y_train[i][5], y_train[i][6],
                y_train[i][9], y_train[i][10], y_train[i][13], y_train[i][14],
                y_train[i][17], y_train[i][18], y_train[i][21], y_train[i][22]
                ])
    
    if arg_mode == 8:
        for i in range(dim_row):
            y_train_1.append([
                y_train[i][0], y_train[i][2], y_train[i][3], y_train[i][5],
                y_train[i][6], y_train[i][8], y_train[i][9], y_train[i][11],
                y_train[i][12], y_train[i][14], y_train[i][15], y_train[i][17]
                ])
            y_train_2.append([
                y_train[i][1], y_train[i][4], y_train[i][7], 
                y_train[i][10], y_train[i][13], y_train[i][16]
                ])

    return y_train_1, y_train_2


def combine_target(y_test_1, y_test_2, *, arg_mode = 7):
    '''
    Combine two target arrays into one.
    C. Wibisono
    07/01 '26
    Parameter(s):
    y_test_1: [arr] target variable 1 #linear
    y_test_2: [arr] target variable 2 #non-linear
    arg_mode: (int) mode of target variable (see get_features_append function).
    Return(s):
    y_train: [arr] target variables
    '''

    y_test = [] #Combined array
    dim1 = len(y_test_1)
    dim2 = len(y_test_2)

    if dim1 != dim2:
        print("Length of arr 1 and 2 has to be equal\n")

    if arg_mode == 7 or arg_mode == 9:
        for i in range(dim1):
            y_test.append([
                y_test_1[i][0], y_test_2[i][0], y_test_2[i][1], y_test_1[i][1],
                y_test_1[i][2], y_test_2[i][2], y_test_2[i][3], y_test_1[i][3],
                y_test_1[i][4], y_test_2[i][4], y_test_2[i][5], y_test_1[i][5],
                y_test_1[i][6], y_test_2[i][6], y_test_2[i][7], y_test_1[i][7],
                y_test_1[i][8], y_test_2[i][8], y_test_2[i][9], y_test_1[i][9],
                y_test_1[i][10], y_test_2[i][10], y_test_2[i][11], y_test_1[i][11]
                ])

    if arg_mode == 8:
        for i in range(dim1):
            y_test.append([
                y_test_1[i][0], y_test_2[i][0], y_test_1[i][1],
                y_test_1[i][2], y_test_2[i][1], y_test_1[i][3],
                y_test_1[i][4], y_test_2[i][2], y_test_1[i][5],
                y_test_1[i][6], y_test_2[i][3], y_test_1[i][7],
                y_test_1[i][8], y_test_2[i][4], y_test_1[i][9],
                y_test_1[i][10], y_test_2[i][5], y_test_1[i][11]
                ])

    return y_test

def rescaled_target_transform(y_test,*,arg_mode = 9):
    '''
    Rescaled transformed target
    C. Wibisono
    06/06 '26
    Parameter(s):
    y_test: [arr] the prediction variables array from the model (After scaling the variable)
    Return(s):
    y_transf: [arr] rescaled target.
    '''

    dim_row = len(y_test)
    dim_column = len(y_test[0])

    y_transf = []
    
    if arg_mode == 9:
        for i in range(dim_row):
            for j in range(0, dim_column, 4):
                #Enforce the negative number of radius to zero:
                if y_test[i][j] < 0.:
                    y_test[i][j] = 0.0
                
                #Convert back to radius:
                y_test[i][j] = (y_test[i][j])**(0.5) 
                #Safety check for preventing division with zero:
                if y_test[i][j] < 0.00001:
                    y_test[i][j] = 0.00001
                    
            for k in range(1, dim_column, 4):
                #Revert scaled angular coordinates to original angular coordinates (cos\theta, sin\theta):
                y_test[i][k] = y_test[i][k] / y_test[i][k-1]
                y_test[i][k+1] = y_test[i][k+1] / y_test[i][k-1]

                #Enforce the trigonometric identity: (sin \theta^2 + cos \theta^2 = 1)
                norm = (((y_test[i][k])**2.)+((y_test[i][k+1])**2.))**0.5

                y_test[i][k] = y_test[i][k]/norm
                y_test[i][k+1] = y_test[i][k+1]/norm
            
            y_transf.append([
                y_test[i][0], y_test[i][1], y_test[i][2], y_test[i][3],
                y_test[i][4], y_test[i][5], y_test[i][6], y_test[i][7],
                y_test[i][8], y_test[i][9], y_test[i][10], y_test[i][11],
                y_test[i][12], y_test[i][13], y_test[i][14], y_test[i][15],
                y_test[i][16], y_test[i][17], y_test[i][18], y_test[i][19],
                y_test[i][20], y_test[i][21], y_test[i][22], y_test[i][23]
                ])

    return y_transf

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


def write_features(fout, y_test, *, coord_transform = 1):
    '''
    Export the result from the model prediction
    C. Wibisono
    04/29 '26
    Parameter(s):
    fout: file output pointer object to store the result 
    y_test: [arr] the prediction variables array from the model
    coord_transform: (int) coordinate transformation of the target. (1: default (no transformation), (2: with transformation)
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
            if coord_transform == 1:
                for i in range(dim):
                    f.write(str(y_test[i][0])+','+str(y_test[i][1])+','+str(y_test[i][2])+','+ \
                            str(y_test[i][3])+','+str(y_test[i][4])+','+str(y_test[i][5])+','+ \
                            str(y_test[i][6])+','+str(y_test[i][7])+','+str(y_test[i][8])+','+ \
                            str(y_test[i][9])+','+str(y_test[i][10])+','+str(y_test[i][11])+'\n')
            if coord_transform == 2:
                for i in range(dim):
                    x0, y0, z0 = cylind_to_cart_theta(y_test[i][0], y_test[i][1], y_test[i][2])
                    x1, y1, z1 = cylind_to_cart_theta(y_test[i][3], y_test[i][4], y_test[i][5])
                    x2, y2, z2 = cylind_to_cart_theta(y_test[i][6], y_test[i][7], y_test[i][8])
                    x3, y3, z3 = cylind_to_cart_theta(y_test[i][9], y_test[i][10], y_test[i][11])
                    f.write(str(x0)+','+str(y0)+','+str(z0)+','+ \
                            str(x1)+','+str(y1)+','+str(y1)+','+ \
                            str(x2)+','+str(y2)+','+str(z2)+','+ \
                            str(x3)+','+str(y3)+','+str(z3)+'\n')

        if dim_column == 18: 
            if coord_transform == 1:
                for i in range(dim):
                    f.write(str(y_test[i][0])+','+str(y_test[i][1])+','+str(y_test[i][2])+','+ \
                            str(y_test[i][3])+','+str(y_test[i][4])+','+str(y_test[i][5])+','+ \
                            str(y_test[i][6])+','+str(y_test[i][7])+','+str(y_test[i][8])+','+ \
                            str(y_test[i][9])+','+str(y_test[i][10])+','+str(y_test[i][11])+','+\
                            str(y_test[i][12])+','+str(y_test[i][13])+','+str(y_test[i][14])+','+\
                            str(y_test[i][15])+','+str(y_test[i][16])+','+str(y_test[i][17])+'\n')
            if coord_transform == 2:
                for i in range(dim):
                    x0, y0, z0 = cylind_to_cart_theta(y_test[i][0], y_test[i][1], y_test[i][2])
                    x1, y1, z1 = cylind_to_cart_theta(y_test[i][3], y_test[i][4], y_test[i][5])
                    x2, y2, z2 = cylind_to_cart_theta(y_test[i][6], y_test[i][7], y_test[i][8])
                    x3, y3, z3 = cylind_to_cart_theta(y_test[i][9], y_test[i][10], y_test[i][11])
                    x4, y4, z4 = cylind_to_cart_theta(y_test[i][12], y_test[i][13], y_test[i][14])
                    x5, y5, z5 = cylind_to_cart_theta(y_test[i][15], y_test[i][16], y_test[i][17])
                    f.write(str(x0)+','+str(y0)+','+str(z0)+','+ \
                            str(x1)+','+str(y1)+','+str(y1)+','+ \
                            str(x2)+','+str(y2)+','+str(z2)+','+ \
                            str(x3)+','+str(y3)+','+str(z3)+','+ \
                            str(x4)+','+str(y4)+','+str(z4)+','+ \
                            str(x5)+','+str(y5)+','+str(z5)+'\n')



        if dim_column == 24:
            for i in range(dim):
                x0, y0, z0 = cylind_to_cart(y_test[i][0], y_test[i][1], y_test[i][2], y_test[i][3])
                x1, y1, z1 = cylind_to_cart(y_test[i][4], y_test[i][5], y_test[i][6], y_test[i][7])
                x2, y2, z2 = cylind_to_cart(y_test[i][8], y_test[i][9], y_test[i][10], y_test[i][11])
                x3, y3, z3 = cylind_to_cart(y_test[i][12], y_test[i][13], y_test[i][14], y_test[i][15])
                x4, y4, z4 = cylind_to_cart(y_test[i][16], y_test[i][17], y_test[i][18], y_test[i][19])
                x5, y5, z5 = cylind_to_cart(y_test[i][20], y_test[i][21], y_test[i][22], y_test[i][23])
                f.write(str(x0)+','+str(y0)+','+str(z0)+','+ \
                        str(x1)+','+str(y1)+','+str(z1)+','+ \
                        str(x2)+','+str(y2)+','+str(z2)+','+ \
                        str(x3)+','+str(y3)+','+str(z3)+','+\
                        str(x4)+','+str(y4)+','+str(z4)+','+\
                        str(x5)+','+str(y5)+','+str(z5)+'\n')


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

    if dim_column == 18:
        ax.scatter(coords[:,0],coords[:,1],coords[:,2],c='blue',marker='o',s=0.001) #First Hit
        ax.scatter(coords[:,3],coords[:,4],coords[:,5],c='blue',marker='o',s=0.001) #Mid Hit
        ax.scatter(coords[:,6],coords[:,7],coords[:,8],c='blue',marker='o',s=0.001) #Last Hit
        ax.scatter(coords[:,9],coords[:,10],coords[:,11],c='red',marker='o',s=0.001) #First Hit
        ax.scatter(coords[:,12],coords[:,13],coords[:,14],c='red',marker='o',s=0.001) #Mid Hit
        ax.scatter(coords[:,15],coords[:,16],coords[:,17],c='red',marker='o',s=0.001) #Last Hit

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


def get_layer_from_file(f_layer):
    '''
    Extract layers from layer file and convert it into a tuple.
    C. Wibisono
    06/15 '26
    Parameter(s):
    f_layer: file pointer object consisting of nn configuration layer
    Return(s):
    layer: (tuple) number of neuron(s) per each ith layer
    '''

    temp_arr = []
    
    with open(f_layer, mode='r') as fin:
        #Read the header:
        fin.readline()
        

        while(1):
            line = fin.readline()
            if line == '':
                break
            else:
                try:
                    temp = int(line)
                    temp_arr.append(temp)
                except ValueError:
                    return "Layer file is not valid. Remove any row that does not contain an integer."

    layer =  tuple(temp_arr)

    return layer


def calculate_mse_loss(X_valid, y_valid, f_model):
    '''
    Calculate the mean squared error from the model stored in f_model.
    C. Wibisono
    06/17 '26
    Parameter(s):
    X_valid: [arr] independent features of the validation data set.
    y_valid: [arr] target variable(s) of the validation data set.
    f_model: [obj] file where model is stored
    Return(s):
    val_loss: (float) mean squared error from the difference between validated data and model prediction
    '''

    from sklearn.metrics import mean_squared_error

    #Load the model:
    model = joblib.load(f_model)

    #Predict the validation data set from the loaded model:
    y_predict = model.predict(X_valid)

    #Calculate the MSE from the validation data over the model prediction:
    val_loss = mean_squared_error(y_valid, y_predict)

    return val_loss

def convergence_check(current_loss, best_loss, flag_counter, tolerance = 1e-4, patience = 5):
    '''
    Check the model convergence based on validation loss data.
    06/17 '26
    Parameter(s):
    current_loss: (float) mean squared error from the difference between validated data and model prediction.
    best_loss: (float) the lowest validation loss recorded.
    flag_counter: (int) current consecutive iteration without improvement relative to previous iteration
    patience: (int) Max iteration to wait for improvement before stopping.
    Return(s):
    is_converged: (int) 1 (model converged) 0 (otherwise)
    best_loss: (float) updated validation loss.
    flag_counter: (int) updated flag counter
    '''

    if current_loss < (best_loss - tolerance):
        best_loss = current_loss
        flag_counter = 0
    else:
        flag_counter = flag_counter + 1

    if flag_counter >= patience:
        is_converged = 1
    else:
        is_converged = 0

    return is_converged, best_loss, flag_counter


