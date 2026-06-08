#!/usr/bin/env python3
import numpy as np
import math


def is_valid_event(data):
    '''
    Check whether a given event is valid based on the hits recorded on the 4 detectors:
    C. Wibisono
    10/ 20 '25
    '''

    key = list(data.keys())[0]
    det_num = {0,1,2,3}

    #First Layer check if hits records all detector numbers:
    result = det_num.issubset(set(data[key][2]))

    temp_arr = {}
    dim = len(data[key][0])
    for i in range(dim):
        parent_ID = data[key][0][i]
        #print(parent_ID)
        if parent_ID in temp_arr.keys():
            temp_arr[parent_ID][0].append(data[key][2][i])
            temp_arr[parent_ID][1].append(data[key][3][i])
            temp_arr[parent_ID][2].append(data[key][4][i])
            temp_arr[parent_ID][3].append(data[key][5][i])

        else:
            temp_arr[parent_ID] = [[data[key][2][i]],[data[key][3][i]],[data[key][4][i]],[data[key][5][i]]]
    
    #Second order check if hits records all detector numbers and they are all muons:
    #result_b=det_num.issubset(set(temp_arr[0][0]))
    if result == True:
        if 0 in temp_arr.keys():
            result_b = det_num.issubset(set(temp_arr[0][0]))
            if result_b == True:
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def generate_input_poca(data):
    '''
    Generate PoCA input. This function can be used 
    only if is_valid_event(data) returns True.
    C. Wibisono
    10/20 '25
    '''
    key = list(data.keys())[0]
    
    temp_arr = {}
    dim = len(data[key][0])
    for i in range(dim):
        parent_ID = data[key][0][i]
        #print(parent_ID)
        if parent_ID in temp_arr.keys():
            temp_arr[parent_ID][0].append(data[key][2][i])
            temp_arr[parent_ID][1].append(data[key][3][i])
            temp_arr[parent_ID][2].append(data[key][4][i])
            temp_arr[parent_ID][3].append(data[key][5][i])

        else:
            temp_arr[parent_ID] = [[data[key][2][i]],[data[key][3][i]],[data[key][4][i]],[data[key][5][i]]]

    

    mult = len(temp_arr[0][0])
    flag_0 = 0
    flag_1 = 0
    flag_2 = 0
    flag_3 = 0

    for j in range(mult):
        if temp_arr[0][0][j] == 0:
            flag_0 = flag_0 + 1
            if flag_0 == 1:
                p1 = np.array([temp_arr[0][1][j], temp_arr[0][2][j], temp_arr[0][3][j]])

        if temp_arr[0][0][j] == 1:
            flag_1 = flag_1 + 1
            if flag_1 == 1:
                p1_b = np.array([temp_arr[0][1][j], temp_arr[0][2][j], temp_arr[0][3][j]])
       
        if temp_arr[0][0][j] == 2:
            flag_2 = flag_2 + 1
            if flag_2 == 1:
                p2 = np.array([temp_arr[0][1][j], temp_arr[0][2][j], temp_arr[0][3][j]])

        if temp_arr[0][0][j] == 3:
            flag_3 = flag_3 + 1
            if flag_3 == 1:
                p2_b = np.array([temp_arr[0][1][j], temp_arr[0][2][j], temp_arr[0][3][j]])
    


    #Generate Direction Vector:
    num_v1 = p1_b - p1
    denum_v1 = np.linalg.norm(num_v1)
    v1 = num_v1/denum_v1

    num_v2 = p2_b - p2
    denum_v2 = np.linalg.norm(num_v2)
    v2 = num_v2/denum_v2

    return p1, v1, p2, v2


#-----------------------------
# Helper: compute PoCA between two tracks
# -----------------------------
def poca_point(p1, v1, p2, v2):
    """
    Compute the midpoint between two lines at their closest approach.
    p1, p2: starting points (3D vectors)
    v1, v2: normalized direction vectors
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    w0 = p1 - p2
    a = np.dot(v1, v1)
    b = np.dot(v1, v2)
    c = np.dot(v2, v2)
    d = np.dot(v1, w0)
    e = np.dot(v2, w0)

    denom = a * c - b * b
    if abs(denom) < 1e-10:
        # Lines are (almost) parallel → return midpoint of reference points
        return 0.5 * (p1 + p2)

    t1 = (b * e - c * d) / denom
    t2 = (a * e - b * d) / denom

    point1 = p1 + t1 * v1
    point2 = p2 + t2 * v2
    poca = 0.5 * (point1 + point2)
    return poca

def scatt_angle(v1, v2):
    """
    Compute the scattering angle between incoming and outgoing muon vector.
    C. Wibisono
    10/22 '25
    Parameter(s):
    p1, p2: starting points (3D vectors)
    v1, v2: normalized direction vectors
    """
    theta_deg = np.rad2deg(np.arccos(np.dot(v1,v2)))
    
    return theta_deg

def poca_point_output(fout,poca_data):
    '''
    Write the poca_points into a file.
    C. Wibisono
    10/20 '25
    Parameter(s):
    fout: file output pointer object
    poca_data: (3D vector) poca points
    '''
    fout.write(str(poca_data[0])+'\t'+str(poca_data[1])+'\t'+str(poca_data[2])+'\n')

def poca_point_output_with_angle(fout, poca_data, theta):
    '''
    Write the poca_points into a file with scattering angle.
    C. Wibisono
    10/20 '25
    Parameter(s):
    fout: file output pointer object
    poca_data: (3D vector) poca points
    theta: scattering angle
    '''
    fout.write(str(poca_data[0])+'\t'+str(poca_data[1])+'\t'+str(poca_data[2])+'\t'+str(theta)+'\n')

def poca_point_output_with_angle_beam_en(fout, poca_data, theta, E_beam):
    '''
    Write the poca_points into a file with scattering angle.
    C. Wibisono
    10/20 '25
    Parameter(s):
    fout: file output pointer object
    poca_data: (3D vector) poca points
    theta: scattering angle
    E_beam: incoming beam energy
    '''
    fout.write(str(poca_data[0])+'\t'+str(poca_data[1])+'\t'+str(poca_data[2])+'\t'+str(theta)+'\t'+str(E_beam)+'\n')

def get_beam_energy(fdata):
    '''
    Transform the incoming beam energy data into a dict data
    C. Wibisono
    10/28 '25
    Parameter(s):
    fdata: file input pointer object
    Return(s):
    data: dictionary of event (key) and beam energy (value)
    '''
    data = {}
    with open(fdata, mode='r') as f:
        while(1):
            line = f.readline()
        
            if line == '':
                break
            if line.find('#') == -1:
                row = line.split(',')
                data[row[0]] = float(row[1].split('\n')[0])


        return data
# -----------------------------
# Step 2: generate synthetic muon tracks
# -----------------------------
'''
np.random.seed(42)

N = 200  # number of muon events
z_top = 100.0
z_bottom = 0.0

poca_points = []
for _ in range(N):
    # Incoming track: starts from top detector
    x1, y1 = np.random.uniform(-50, 50, 2)
    p1 = np.array([x1, y1, z_top])

    # Small angular deviation downward
    theta_in = np.random.normal(0, 0.02)  # radians
    phi_in = np.random.uniform(0, 2*np.pi)
    v1 = np.array([np.sin(theta_in)*np.cos(phi_in),
                   np.sin(theta_in)*np.sin(phi_in),
                   -np.cos(theta_in)])

    # Outgoing track: starts from bottom detector
    x2, y2 = np.random.uniform(-50, 50, 2)
    p2 = np.array([x2, y2, z_bottom])

    # Add scattering (simulate object near z=50)
    scatter_theta = np.random.normal(0, 0.1)
    scatter_phi = np.random.uniform(0, 2*np.pi)
    v2 = np.array([np.sin(scatter_theta)*np.cos(scatter_phi),
                   np.sin(scatter_theta)*np.sin(scatter_phi),
                   np.cos(scatter_theta)])

    poca = poca_point(p1, v1, p2, v2)
    poca_points.append(poca)

poca_points = np.array(poca_points)

# -----------------------------
# Step 3: visualize the PoCA points
# -----------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(poca_points[:, 0], poca_points[:, 1], poca_points[:, 2],
           c=poca_points[:, 2], cmap='viridis', s=15)

ax.set_xlabel('X [cm]')
ax.set_ylabel('Y [cm]')
ax.set_zlabel('Z [cm]')
ax.set_title('Simulated PoCA Points from Muon Tracks')
ax.set_xlim(-60, 60)
ax.set_ylim(-60, 60)
ax.set_zlim(0, 100)
plt.show()
'''


def obj_dict(fin):
    '''
    Generate dictionary for an object being probed
    C. Wibisono
    04/27 '26
    Parameter(s):
    fin: file pointer input file
    data: dictionary of object being probed position based on event ID
    '''

    data = {}
    with open(fin, mode='r') as f:

        while(1):
            line = f.readline()
            if line == '':
                break

            if line.find('#') == -1:
                row = line.split(',')
                
                data[row[0]] = [int(row[1]), int(row[2]), float(row[3]), float(row[4]), float(row[5].split('\n')[0])]


    return data

def obj_dict_append(fin):
    '''
    Generate dictionary for an object being probed
    allowed multiple records with the same primary key
    C. Wibisono
    05/18 '26
    Parameter(s):
    fin: file pointer input file
    data: dictionary of object being probed position based on event ID
    '''

    data = {}
    with open(fin, mode='r') as f:

        while(1):
            line = f.readline()
            if line == '':
                break

            if line.find('#') == -1:
                row = line.split(',')

                if row[0] in data.keys():
                    data[row[0]][0].append(int(row[1]))
                    data[row[0]][1].append(int(row[2]))
                    data[row[0]][2].append(float(row[3]))
                    data[row[0]][3].append(float(row[4]))
                    data[row[0]][4].append(float(row[5].split('\n')[0]))

                else:
                    data[row[0]] = [[int(row[1])],[int(row[2])],[float(row[3])],[float(row[4])],[float(row[5].split('\n')[0])]]



    return data



def generate_input_scint(data):
    '''
    Generate Scintillators input. This function can be used 
    only if is_valid_event(data) returns True.
    C. Wibisono
    04/27 '26
    '''
    key = list(data.keys())[0]
    
    temp_arr = {}
    dim = len(data[key][0])
    for i in range(dim):
        parent_ID = data[key][0][i]
        #print(parent_ID)
        if parent_ID in temp_arr.keys():
            temp_arr[parent_ID][0].append(data[key][2][i])
            temp_arr[parent_ID][1].append(data[key][3][i])
            temp_arr[parent_ID][2].append(data[key][4][i])
            temp_arr[parent_ID][3].append(data[key][5][i])

        else:
            temp_arr[parent_ID] = [[data[key][2][i]],[data[key][3][i]],[data[key][4][i]],[data[key][5][i]]]

    

    mult = len(temp_arr[0][0])
    flag_0 = 0
    flag_1 = 0
    flag_2 = 0
    flag_3 = 0

    for j in range(mult):
        if temp_arr[0][0][j] == 0:
            flag_0 = flag_0 + 1
            if flag_0 == 1:
                p1 = np.array([temp_arr[0][1][j], temp_arr[0][2][j], temp_arr[0][3][j]])

        if temp_arr[0][0][j] == 1:
            flag_1 = flag_1 + 1
            if flag_1 == 1:
                p1_b = np.array([temp_arr[0][1][j], temp_arr[0][2][j], temp_arr[0][3][j]])
       
        if temp_arr[0][0][j] == 2:
            flag_2 = flag_2 + 1
            if flag_2 == 1:
                p2 = np.array([temp_arr[0][1][j], temp_arr[0][2][j], temp_arr[0][3][j]])

        if temp_arr[0][0][j] == 3:
            flag_3 = flag_3 + 1
            if flag_3 == 1:
                p2_b = np.array([temp_arr[0][1][j], temp_arr[0][2][j], temp_arr[0][3][j]])
    


    #Generate Direction Vector:
    '''
    num_v1 = p1_b - p1
    denum_v1 = np.linalg.norm(num_v1)
    v1 = num_v1/denum_v1

    num_v2 = p2_b - p2
    denum_v2 = np.linalg.norm(num_v2)
    v2 = num_v2/denum_v2
    '''

    return p1, p1_b, p2, p2_b


def write_header(fout, mode):
    '''
    Write Header to identify correlated file types
    C. Wibisono
    05/18 '26
    Parameter(s):
    fout: correlation file output pointer object
    mode: (int) writing mode (1) only the first/last/average hits of the probed objects to be retrieved, (2): otherwise
    '''
    if mode == 1:
        fout.write('#ev_ID'+','+'#Scint_0_pos(x,y,z)'+','+'#Scint_1_pos(x,y,z)'+','+'#Scint_2_pos(x,y,z)'+','+'#Scint_3_pos(x,y,z)'+','+'Pipe_Pos(x,y,z)'+','+'Scaling_Pos(x,y,z)'+'\n')
    
    if mode == 2:
        fout.write('#ev_ID'+','+'#Pos_(x,y,z)'+','+'Object_Type'+'\n')



def corr_file(fout, scint_0, scint_1, scint_2, scint_3, pipe_data, scaling_data, key):
    '''
    Generate correlation file output consists of
    spatial positions in scintillators, pipe, and scaling
    C. Wibisono
    04/27 '26
    Parameter(s):
    fout: correlation file output pointer object
    scint_0: (dict) scintillators data 1st layer
    scint_1: (dict) scintillators data 2nd layer
    scint_2: (dict) scintillators data 3rd layer
    scint_3: (dict) scintillators data 4th layer
    pipe_data: (dict) pipe data
    scaling_data: (dict) scaling data
    key: str primary key (event ID)
    '''

    fout.write(key+','+str(scint_0[0])+','+str(scint_0[1])+','+str(scint_0[2])+','+str(scint_1[0])+','+str(scint_1[1])+','+str(scint_1[2])+','+str(scint_2[0])+','+str(scint_2[1])+','+str(scint_2[2])+','+str(scint_3[0])+','+str(scint_3[1])+','+str(scint_3[2])+','+str(pipe_data[key][2])+','+str(pipe_data[key][3])+','+str(pipe_data[key][4])+','+str(scaling_data[key][2])+','+str(scaling_data[key][3])+','+str(scaling_data[key][4])+'\n')




def corr_file_append(fout, scint_0, scint_1, scint_2, scint_3, pipe_data, scaling_data, key):
    '''
    Generate correlation file output consists of
    spatial positions in scintillators, pipe, and scaling.
    The file can allow multiple scattering points for the pipe and scaling.
    C. Wibisono
    05/18 '26
    Parameter(s):
    fout: correlation file output pointer object
    scint_0: (dict) scintillators data 1st layer
    scint_1: (dict) scintillators data 2nd layer
    scint_2: (dict) scintillators data 3rd layer
    scint_3: (dict) scintillators data 4th layer
    pipe_data: (dict) pipe data
    scaling_data: (dict) scaling data
    key: str primary key (event ID)
    '''

    pipe_dim = len(pipe_data[key][0])
    scaling_dim = len(scaling_data[key][0])

    fout.write(key+','+str(scint_0[0])+','+str(scint_0[1])+','+str(scint_0[2])+','+'0'+'\n')
    fout.write(key+','+str(scint_1[0])+','+str(scint_1[1])+','+str(scint_1[2])+','+'1'+'\n')
    fout.write(key+','+str(scint_2[0])+','+str(scint_2[1])+','+str(scint_2[2])+','+'2'+'\n')
    fout.write(key+','+str(scint_3[0])+','+str(scint_3[1])+','+str(scint_3[2])+','+'3'+'\n')
    for i in range(pipe_dim):
        fout.write(key+','+str(pipe_data[key][2][i])+','+str(pipe_data[key][3][i])+','+str(pipe_data[key][4][i])+','+'4'+'\n')
    for j in range(scaling_dim):
        fout.write(key+','+str(scaling_data[key][2][j])+','+str(scaling_data[key][3][j])+','+str(scaling_data[key][4][j])+','+'5'+'\n')


def get_radius(fin, mode):
    '''
    Get the maximum radius of the objects
    from the correlated file
    C. Wibisono
    05/20 '26:
    Parameter(s):
    fin: correlation file pointer object
    mode: running_mode(1, 2, 3 or 4)
    
    Return(s):
    obj: [dict] maximum and minimum radius of each object
    '''

    obj = {}
    r_pipe_min = 10000
    r_pipe_max = 0
    r_scaling_min = 10000
    r_scaling_max = 0

    with open(fin, mode = 'r') as f:
        #Read the header:
        f.readline()

        if (mode == 1 or mode == 2 or mode == 4):
            while(1):
                line = f.readline()
                if line == '':
                    break
                else:
                    row = line.split(',')
                    
                    x_pipe = float(row[13])
                    y_pipe = float(row[14])
                    z_pipe = float(row[15])
                    rad_temp_pipe = math.sqrt((x_pipe**2.)+(z_pipe**2.))

                    if rad_temp_pipe > r_pipe_max:
                        r_pipe_max = rad_temp_pipe
                    
                    if rad_temp_pipe < r_pipe_min:
                        r_pipe_min = rad_temp_pipe

                    x_scaling = float(row[16])
                    y_scaling = float(row[17])
                    z_scaling = float(row[18])

                    rad_temp_scaling = math.sqrt((x_scaling**2.)+(z_scaling**2.))

                    if rad_temp_scaling > r_scaling_max:
                        r_scaling_max = rad_temp_scaling

                    if rad_temp_scaling < r_scaling_min:
                        r_scaling_min = rad_temp_scaling

        if mode == 3:
            while(1):
                line = f.readline()
                if line == '':
                    break
                else:
                    row = line.split(',')
                    
                    obj_type = int(row[4])
                    
                    if obj_type == 4:
                        x_pipe = float(row[1])
                        y_pipe = float(row[2])
                        z_pipe = float(row[3])
                        rad_temp_pipe = math.sqrt((x_pipe**2.)+(z_pipe**2.))

                        if rad_temp_pipe > r_pipe_max:
                            r_pipe_max = rad_temp_pipe
                    
                        if rad_temp_pipe < r_pipe_min:
                            r_pipe_min = rad_temp_pipe

                    if obj_type == 5:
                        x_scaling = float(row[1])
                        y_scaling = float(row[2])
                        z_scaling = float(row[3])

                        rad_temp_scaling = math.sqrt((x_scaling**2.)+(z_scaling**2.))

                        if rad_temp_scaling > r_scaling_max:
                            r_scaling_max = rad_temp_scaling

                        if rad_temp_scaling < r_scaling_min:
                            r_scaling_min = rad_temp_scaling


        
        obj['pipe'] = [r_pipe_min/10., r_pipe_max/10.] #in cm
        obj['scaling'] = [r_scaling_min/10., r_scaling_max/10.] #in cm

    return obj


def get_hits_number(file_in, file_out):
    '''
    Get the number of hits per event for all hits retrieved on the objects
    C. Wibisono
    06/04 '26
    Parameter(s):
    file_in: correlated fileinput pointer object
    file_out: file output pointer object
    '''

    with open(file_in, mode='r') as fin:

        #Read the header:
        fin.readline()

        data = {}
        with open(file_out, mode ='w') as fout:
            fout.write('#event_id'+','+'Num_Hits_Pipe'+','+'Num_Hits_Scaling'+'\n')

            while(1):
                line = fin.readline()
                if line == '':
                    break

                else:
                    row = line.split(',')
                    obj_type = int(row[4].split('\n')[0])

                    if row[0] in data.keys():
                        if obj_type == 4:
                            pipe_hits = pipe_hits + 1

                        if obj_type == 5:
                            scaling_hits = scaling_hits + 1

                    else:
                        if not data:
                            prev_id = row[0]
                            pipe_hits = 0
                            scaling_hits = 0

                            data[row[0]] = []

                        else: #count the number of hits from previous event key
                            data[prev_id].append(pipe_hits)
                            data[prev_id].append(scaling_hits)
                            fout.write(str(prev_id)+','+str(data[prev_id][0])+','+str(data[prev_id][1])+'\n')

                            del data
                            data = {}
                            data[row[0]] = []
                            prev_id = row[0]
                            pipe_hits = 0
                            scaling_hits = 0


            data[prev_id].append(pipe_hits)
            data[prev_id].append(scaling_hits)
            fout.write(str(prev_id)+','+str(data[prev_id][0])+','+str(data[prev_id][1])+'\n')
                
            del data


def get_hits_number_in_out(file_in, file_out):
    '''
    Get the number of hits per event for all hits retrieved on the objects distinguished by incoming and outgoing muons across the objects
    C. Wibisono
    06/08 '26
    Parameter(s):
    file_in: correlated fileinput pointer object
    file_out: file output pointer object
    '''

    with open(file_in, mode='r') as fin:

        #Read the header:
        fin.readline()

        data = {}
        with open(file_out, mode ='w') as fout:
            fout.write('#event_id'+','+'Num_Hits_Pipe_In'+','+'Num_Hits_Pipe_Out'+','+'Num_Hits_Scaling_In'+','+'Num_Hits_Scaling_Out'+'\n')

            while(1):
                line = fin.readline()
                if line == '':
                    break

                else:
                    row = line.split(',')
                    obj_type = int(row[4].split('\n')[0])
                    pos_x = float(row[1])
                    pos_y = float(row[2])
                    pos_z = float(row[3])

                    if row[0] in data.keys():
                        if obj_type == 4:
                            if pos_z < 0:
                                pipe_hits_out = pipe_hits_out + 1
                            if pos_z > 0:
                                pipe_hits_in = pipe_hits_in + 1

                        if obj_type == 5:
                            if pos_z > 0:
                                scaling_hits_in = scaling_hits_in + 1
                            if pos_z < 0:
                                scaling_hits_out = scaling_hits_out + 1

                    else:
                        if not data:
                            prev_id = row[0]
                            pipe_hits_in = 0
                            pipe_hits_out = 0
                            scaling_hits_in = 0
                            scaling_hits_out = 0

                            data[row[0]] = []

                        else: #count the number of hits from previous event key
                            data[prev_id].append(pipe_hits_in)
                            data[prev_id].append(pipe_hits_out)
                            data[prev_id].append(scaling_hits_in)
                            data[prev_id].append(scaling_hits_out)
                            fout.write(str(prev_id)+','+str(data[prev_id][0])+','+str(data[prev_id][1])+ \
                                    ','+str(data[prev_id][2])+','+str(data[prev_id][3])+'\n')

                            del data
                            data = {}
                            data[row[0]] = []
                            prev_id = row[0]
                            pipe_hits_in = 0
                            pipe_hits_out = 0
                            scaling_hits_in = 0
                            scaling_hits_out = 0


            data[prev_id].append(pipe_hits_in)
            data[prev_id].append(pipe_hits_out)
            data[prev_id].append(scaling_hits_in)
            data[prev_id].append(scaling_hits_out)
            fout.write(str(prev_id)+','+str(data[prev_id][0])+','+str(data[prev_id][1])+ \
                    ','+str(data[prev_id][2])+','+str(data[prev_id][3])+'\n')
                
            del data



def plot_obj_hits(file_in):
    '''
    Plot the number of hits retrieved from scaling and pipe.
    C. Wibisono
    06/04 '26
    Parameter(s):
    file_in: file input pointer object where the number of hits are stored:
    '''

    import matplotlib.pyplot as plt

    with open(file_in, mode ='r') as f:
        f.readline()
        pipe_hits = []
        scaling_hits = []

        while(1):
            line = f.readline()
            
            if line == '':
                break

            else:
                row = line.split(',')
                pipe_hits.append(int(row[1]))
                scaling_hits.append(int(row[2]))
                        
                        
        fig, ax = plt.subplots(1,2)
        ax[0].hist(pipe_hits)
        ax[1].hist(scaling_hits)
        ax[0].set_title('#Pipe hits')
        ax[1].set_title('#Scaling hits')
        plt.show()



def plot_obj_hits_2D(file_in):
    '''
    Plot the number of hits retrieved from scaling and pipe for the incoming vs outgoing muons.
    C. Wibisono
    06/08 '26
    Parameter(s):
    file_in: file input pointer object where the number of hits are stored:
    '''

    import matplotlib.pyplot as plt

    with open(file_in, mode ='r') as f:
        f.readline()
        pipe_hits_in = []
        pipe_hits_out = []
        scaling_hits_in = []
        scaling_hits_out = []

        while(1):
            line = f.readline()
            
            if line == '':
                break

            else:
                row = line.split(',')
                pipe_hits_in.append(int(row[1]))
                pipe_hits_out.append(int(row[2]))
                scaling_hits_in.append(int(row[3]))
                scaling_hits_out.append(int(row[4]))
                        
                        
        fig, ax = plt.subplots(1,2)
        h1 = ax[0].hist2d(pipe_hits_in, pipe_hits_out, bins = 20, cmap='Blues')
        h2 = ax[1].hist2d(scaling_hits_in, scaling_hits_out, bins = 20, cmap='Reds')
        fig.colorbar(h1[3], ax = ax[0], label='Pipe Counts')
        fig.colorbar(h2[3], ax = ax[1], label='Scaling Counts') 

        ax[0].set_xlabel('#Muon_In')
        ax[0].set_ylabel('#Muon_Out')
        ax[1].set_xlabel('#Muon_In')
        ax[1].set_ylabel('#Muon_Out')

        ax[0].set_title('#Pipe hits')
        ax[1].set_title('#Scaling hits')
        plt.show()
