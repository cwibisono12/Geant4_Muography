#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utilities import is_valid_event
from utilities import generate_input_scint, write_header
from utilities import obj_dict, obj_dict_append
from utilities import corr_file, corr_file_append

def transform_input(file_in, f_pipe, f_scaling, file_out):
    '''
    Extract event data from Geant4 Simulation results and generate the correlated file
    C. Wibisono
    04/'29 '26
    Parameter(s):
    file_in: file pointer of the Geant4 output detector_hits file
    f_pipe: file pointer of the pipe hit
    f_scaling: file pointer of the scaling hit
    f_out: file pointer  consisting of all correlated hits
    '''

    data ={}
    pipe = obj_dict(f_pipe)
    scaling = obj_dict(f_scaling)
    
    count = 0
    while(1):
        line = file_in.readline()
        if line == '':
            break
        
        if line.find('#') == -1:
            row = line.split(',')
            if row[0] in data.keys():
                data[row[0]][0].append(int(row[1]))
                data[row[0]][1].append(int(row[2]))
                data[row[0]][2].append(int(row[3]))
                data[row[0]][3].append(float(row[4]))
                data[row[0]][4].append(float(row[5]))
                data[row[0]][5].append(float(row[6].split('\n')[0]))
            else:
                if not data:
                    data[row[0]] = [[int(row[1])],[int(row[2])],[int(row[3])],[float(row[4])],[float(row[5])],[float(row[6].split('\n')[0])]]
                else:
                    #print(data)

                    #Process event:
                    flag = is_valid_event(data)
                    #print(flag)

                    if flag == True:
                        scint_key = list(data.keys())[0]
                        if scint_key in pipe.keys():
                            if scint_key in scaling.keys():
                                count = count + 1
                                scint0, scint1, scint2, scint3 = generate_input_scint(data)
                                #print("x0:",data[row[0]][3][0],"y0:",data[row[0]][4][0],"z0:",data[row[0]][5][0],"xobj:",pipe[row[0]][2],"yobj:",pipe[row[0]][3],"zobj:",pipe[row[0]][4],"xobj2:",scaling[row[0]][2],"yobj2:",scaling[row[0]][3],"zobj2:",scaling[row[0]][4])
                                print(scint_key,scint0[0],scint0[1],scint0[2],pipe[scint_key][2],pipe[scint_key][3],pipe[scint_key][4])
                                corr_file(file_out, scint0, scint1, scint2, scint3, pipe, scaling, scint_key)

                    #Store event data into a dictionary:
                    del data
                    data = {}
                    data[row[0]] = [[int(row[1])],[int(row[2])],[int(row[3])],[float(row[4])],[float(row[5])],[float(row[6].split('\n')[0])]]
    
    #Process the last event:
    #print(data)
    flag = is_valid_event(data)
    #print(flag)

    if flag == True:
        scint_key = list(data.keys())[0]
        if scint_key in pipe.keys():
            if scint_key in scaling.keys():
                count = count + 1
                scint0, scint1, scint2, scint3 = generate_input_scint(data)
                #print("x0:",data[row[0]][3][0],"y0:",data[row[0]][4][0],"z0:",data[row[0]][5][0],"xobj:",pipe[row[0]][2],"yobj:",pipe[row[0]][3],"zobj:",pipe[row[0]][4],"xobj2:",scaling[row[0]][2],"yobj2:",scaling[row[0]][3],"zobj2:",scaling[row[0]][4])
                print(row[0],scint0[0],scint0[1],scint0[2],scint0[3],pipe[scint_key][2],pipe[scint_key][3],pipe[scint_key][4])
                corr_file(file_out, scint0, scint1, scint2, scint3, pipe, scaling, scint_key)

    print(count)


def transform_input_append(file_in, f_pipe, f_scaling, file_out):
    '''
    Extract event data from Geant4 Simulation results and generate the correlated file
    Allow multiple scattering points of the pipe and scalling  to be recorded in a correlated file
    C. Wibisono
    05/'18 '26
    Parameter(s):
    file_in: file pointer of the Geant4 output detector_hits file
    f_pipe: file pointer of the pipe hit
    f_scaling: file pointer of the scaling hit
    f_out: file pointer  consisting of all correlated hits
    '''

    data ={}
    pipe = obj_dict_append(f_pipe)
    scaling = obj_dict_append(f_scaling)
    
    count = 0
    while(1):
        line = file_in.readline()
        if line == '':
            break
        
        if line.find('#') == -1:
            row = line.split(',')
            if row[0] in data.keys():
                data[row[0]][0].append(int(row[1]))
                data[row[0]][1].append(int(row[2]))
                data[row[0]][2].append(int(row[3]))
                data[row[0]][3].append(float(row[4]))
                data[row[0]][4].append(float(row[5]))
                data[row[0]][5].append(float(row[6].split('\n')[0]))
            else:
                if not data:
                    data[row[0]] = [[int(row[1])],[int(row[2])],[int(row[3])],[float(row[4])],[float(row[5])],[float(row[6].split('\n')[0])]]
                else:
                    #print(data)

                    #Process event:
                    flag = is_valid_event(data)
                    #print(flag)

                    if flag == True:
                        scint_key = list(data.keys())[0]
                        if scint_key in pipe.keys():
                            if scint_key in scaling.keys():
                                count = count + 1
                                scint0, scint1, scint2, scint3 = generate_input_scint(data)
                                #print("x0:",data[row[0]][3][0],"y0:",data[row[0]][4][0],"z0:",data[row[0]][5][0],"xobj:",pipe[row[0]][2],"yobj:",pipe[row[0]][3],"zobj:",pipe[row[0]][4],"xobj2:",scaling[row[0]][2],"yobj2:",scaling[row[0]][3],"zobj2:",scaling[row[0]][4])
                                print(scint_key,scint0[0],scint0[1],scint0[2],pipe[scint_key][2],pipe[scint_key][3],pipe[scint_key][4])
                                corr_file_append(file_out, scint0, scint1, scint2, scint3, pipe, scaling, scint_key)

                    #Store event data into a dictionary:
                    del data
                    data = {}
                    data[row[0]] = [[int(row[1])],[int(row[2])],[int(row[3])],[float(row[4])],[float(row[5])],[float(row[6].split('\n')[0])]]
    
    #Process the last event:
    #print(data)
    flag = is_valid_event(data)
    #print(flag)

    if flag == True:
        scint_key = list(data.keys())[0]
        if scint_key in pipe.keys():
            if scint_key in scaling.keys():
                count = count + 1
                scint0, scint1, scint2, scint3 = generate_input_scint(data)
                #print("x0:",data[row[0]][3][0],"y0:",data[row[0]][4][0],"z0:",data[row[0]][5][0],"xobj:",pipe[row[0]][2],"yobj:",pipe[row[0]][3],"zobj:",pipe[row[0]][4],"xobj2:",scaling[row[0]][2],"yobj2:",scaling[row[0]][3],"zobj2:",scaling[row[0]][4])
                print(row[0],scint0[0],scint0[1],scint0[2],scint0[3],pipe[scint_key][2],pipe[scint_key][3],pipe[scint_key][4])
                corr_file_append(file_out, scint0, scint1, scint2, scint3, pipe, scaling, scint_key)

    print(count)
    
if __name__ == "__main__":
    import sys
    filein = sys.argv[1] #scintillator file
    filein2 = sys.argv[2] #pipe file
    filein3 = sys.argv[3] #scaling file
    fileout = sys.argv[4] #output file
    with open(fileout, mode = 'w') as fout:
        write_header(fout, 1)
        with open(filein, mode='r') as fin1:
            transform_input(fin1, filein2, filein3, fout)
