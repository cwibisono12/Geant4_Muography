#!/usr/bin/env python3

from nn import get_layer_from_file
from torch import nn
import torch

class MLPRegressor(nn.Module):
    '''
    Build NN architecture inherited from the nn.Module.
    C. Wibisono
    07/08 '26
    activation_function: (str) activation function to be used (eg. 'relu' or 'tanh' default = 'relu').
    '''
    
    def __init__(self, layer, ft_dim, tg_dim, *,  activation_function = 'relu'):
        '''
        Initialize the Constructor
        C. Wibisono
        07/08 '26
        Parameter(s):
        layer: (tuple) number of neurons for each layer ith.
        ft_dim: (int) number of features.
        tg_dim: (int) number of targets.
        '''
        
        super().__init__()
        #Get the layer from layer file:
        self.layer = layer
        dim_layer = len(self.layer)
        
        #Get the feature and target dimensions:
        self.ft_dim = ft_dim
        self.tg_dim = tg_dim

        #Initialize the container network of layers:
        network_list = []
        
        #Setup the Activation Function:
        if activation_function == 'relu':
            self.act_func = nn.ReLU()
        if activation_function == 'tanh':
            self.act_func = nn.Tanh()
        #Set up the layer to be executed when the input data is passed through the object instantiated from this class.:
        network_list.append(nn.Linear(self.ft_dim, self.layer[0]))
        network_list.append(self.act_func)
        
        for i in range(0,dim_layer-1,1):
            network_list.append(nn.Linear(self.layer[i], self.layer[i+1]))
            network_list.append(self.act_func)

        network_list.append(nn.Linear(self.layer[dim_layer-1], self.tg_dim))

        
        self.network = nn.Sequential(*network_list)

    def forward(self, X_train):
        '''
        Define the data flow (neurons) when the input features X_train propagates through the defined layer above.
        C. Wibisono
        07/08 '26
        Parameter(s):
        X_train: [arr] independent input features (data format needs to be pytorch tensor array).
        '''

        return self.network(X_train)

def store_model_torch(model, f_model):
    '''
    Store the model from torch library into a file
    Parameter(s):
    model: (obj) file obje
    '''
    torch.save(model, f_model)

def load_model_torch(f_model):
    '''
    Load the model generated from torch library from file.
    '''
    model = torch.load(f_model, weights_only=False)

    
def train_loop(X_train, y_train, model, loss_fn, optimizer):
    '''
    Train the model
    C. Wibisono
    07/08 '26
    Parameter(s):
    X_train: [arr] independent feature(s). (use torch tensor with .float32 to use this function)
    y_train: [arr] target variables. (use torch tensor with .float32 to use this function)
    model: (obj) torch model.
    loss_fn: (obj) loss function.
    optimizer: (obj) optimizer object.
    '''

    model.train()
    
    
    #Compute the prediction and loss:
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    ave_train_loss = loss.item()
    
    return ave_train_loss

def test_loop(X_test, y_test, model, loss_fn):
    '''
    Evaluate the model over the validation data.
    C. Wibisono
    07/08 '26
    Parameter(s):
    X_test: [arr] independent feature(s). (use torch tensor with .float32 to use this function)
    y_test: [arr] target variables. (use torch tensor with .float32 to use this function)
    model: (obj) torch model.
    loss_fn: (obj) loss function.
    optimizer: (obj) optimizer object.
    '''
    
    model.eval()

    with torch.no_grad():
        y_pred = model(X_test)
        test_loss = loss_fn(y_pred, y_test)

        ave_test_loss = test_loss.item()
    
    return ave_test_loss

def get_model_score_from_file(X_test, y_test, model_param, model_file):
    '''
    Calculate the model score, given the model instantiation and model_file consist of weight
    C. Wibisono
    07/09 '26
    Parameter(s):
    X_test: [arr] independent feature(s). (use torch tensor with .float32 to use this function)
    y_test: [arr] independent feature(s). (use torch tensor with .float32 to use this function)
    model_param: (obj) torch model
    model_file: (obj) file pointer of object to store model weight.
    '''

    from torcheval.metrics.functional import r2_score
    
    #Load the weight from model_file:
    model_param.load_state_dict(torch.load(model_file, weights_only=True))

    model_param.eval()

    with torch.no_grad():
        y_pred = model_param(X_test)

    score_tensor = r2_score(y_pred.flatten().float(), y_test.flatten().float())

    score = score_tensor.item()

    return score

def predict_outcome_from_file(X_test, model_param, model_file):
    '''
    Get the result of the model based on the trained model in model_file
    C. Wibisono
    07/09 '26
    Parameter(s):
    X_test: [arr] independent feature(s). (use torch tensor with .float32 to use this function)
    model_param: (obj) torch model
    model_file: (obj) file pointer of object to store model weight.
    Return(s):
    y_test: [arr] model prediction.
    '''
    

    #Load the weight from model_file:
    model_param.load_state_dict(torch.load(model_file, weights_only = True))

    model_param.eval()

    with torch.no_grad():
        y_pred = model_param(X_test)

    y_test = y_pred.detach().numpy()

    return y_test
