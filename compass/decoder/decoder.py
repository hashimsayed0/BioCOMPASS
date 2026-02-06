import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ModuleList, ReLU, Dropout
from copy import deepcopy 
import pandas as pd
import numpy as np



import random
def fixseed(seed=42): 
    np.random.seed(seed)  
    random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.cuda.manual_seed(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




class ClassDecoder(nn.Module):
    def __init__(self, input_dim=32, dense_layers = [], out_dim = 2, dropout_p = 0.0, batch_norms = True, seed = 42):
        '''
        classification
        '''
        super(ClassDecoder,self).__init__()
        self.seed = seed
        fixseed(seed = seed)
        
        ## Input
        self.input_norm = torch.nn.BatchNorm1d(input_dim)

        _dense_layers = [input_dim]
        _dense_layers.extend(dense_layers)
        self._dense_layers = _dense_layers
        self.batch_norms = batch_norms
        
        ## Dense
        self.lins = ModuleList()
        for i in range(len(_dense_layers)-1):
            lin = Linear(_dense_layers[i], _dense_layers[i+1])
            self.lins.append(lin)

        ## Batchnorm
        self._batch_norms = ModuleList()
        for i in range(len(_dense_layers)-1):
            self._batch_norms.append(deepcopy(torch.nn.BatchNorm1d(_dense_layers[i+1])))

        ## Dropout
        self.dropout = nn.Dropout(dropout_p)

        # Output layer
        last_hidden = _dense_layers[-1]
        self.out = Linear(last_hidden, out_dim)
        self.softmax = nn.Softmax(dim=1)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(1.0)))

    def forward(self, x):
        if self.batch_norms & (len(self._batch_norms) == 0):
            x = self.input_norm(x)
        
        for lin, norm in zip(self.lins, self._batch_norms):
            if self.batch_norms:
                x = self.dropout(F.relu(norm(lin(x)), inplace=False))
            else:
                x = self.dropout(F.relu(lin(x), inplace=False))  

        ## dynamic temperature
        temperature = torch.exp(self.log_temperature)
        y = self.softmax(self.out(x) / temperature)
        
        #y = F.log_softmax(self.out(x), dim=1)
        return y
        



class ProtoNetDecoder(nn.Module):
    def __init__(self, input_dim, out_dim = 2, dense_layers = [],  dropout_p = 0.0, batch_norms = True, seed = 42): 
        super(ProtoNetDecoder, self).__init__()

       
        self.seed = seed
        fixseed(seed = seed)

        # Input
        self.input_norm = torch.nn.BatchNorm1d(input_dim)

        _dense_layers = [input_dim]
        _dense_layers.extend(dense_layers)
        self._dense_layers = _dense_layers
        self.batch_norms = batch_norms
        
        ## Dense
        self.lins = ModuleList()
        for i in range(len(_dense_layers)-1):
            lin = Linear(_dense_layers[i], _dense_layers[i+1])
            self.lins.append(lin)

        ## Batchnorm
        self._batch_norms = ModuleList()
        for i in range(len(_dense_layers)-1):
            self._batch_norms.append(deepcopy(torch.nn.BatchNorm1d(_dense_layers[i+1])))

        # Dropout
        self.dropout = nn.Dropout(dropout_p)

        # Out
        last_hidden = _dense_layers[-1]
        # We only define the shape of W and b here without initializing them
        self.W = nn.Parameter(torch.Tensor(out_dim, last_hidden))  # Will be initialized later with class means
        self.b = nn.Parameter(torch.Tensor(out_dim))  # Bias initialized as zeros
        init.normal_(self.W, mean=0.0, std=0.01)
        init.normal_(self.b, mean=0.0, std=0.01)

        #self.temperature = temperature
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.softmax = nn.Softmax(dim=1)

        self.num_classes = out_dim
        self.feature_dim = input_dim

    
    def forward(self, x):
        # Normalize the input features and weights to calculate cosine similarity

        if self.batch_norms & (len(self._batch_norms) == 0):
            x = self.input_norm(x)

        for lin, norm in zip(self.lins, self._batch_norms):
            if self.batch_norms:
                x = self.dropout(F.relu(norm(lin(x)), inplace=False))
            else:
                x = self.dropout(F.relu(lin(x), inplace=False))  


        x_norm = F.normalize(x, p=2, dim=1)
        W_norm = F.normalize(self.W, p=2, dim=1)
        # Calculate the cosine similarity between the input and prototypes
        # Using torch.mm for batch matrix multiplication
        cosine_similarity = torch.mm(x_norm, W_norm.T)
        # Add the bias term
        logits = cosine_similarity + self.b
        # Apply softmax to get probabilities
        temperature = torch.exp(self.log_temperature)
        probabilities = self.softmax(logits / temperature)
        return probabilities

    
    def initialize_parameters(self, support_features, support_labels):
        """
        Initialize the weights (W) using the class means of the support set features and the biases (b) as zeros.
    
        Parameters:
        support_features (torch.Tensor): The features of the support set.
        support_labels (torch.Tensor): The one-hot encoded labels of the support set.
        """
        with torch.no_grad():  # No need to track gradients during initialization
            for i in range(self.num_classes):
                # Calculate the mean feature vector for class i
                class_features = support_features[support_labels[:, i] == 1]
                class_mean = class_features.mean(dim=0)
                # Assign the mean vector to the corresponding row in W
                self.W[i] = class_mean





# Define the Prototypical Network without fine-tuning
class ProtoNetNFTDecoder:

    def __init__(self, temperature = 0.1):
        
        self.prototype_class_map = {'PD':0, 'SD':0,
                                    'PR':1, 'CR':1, 1:1, 0:0, 
                                    'R':1, 'NR':0}

        self.temperature = temperature
        

    def fit(self, support_set):
        '''
        support_set: the last column is the RECIST label column
        '''
        self.label_col = support_set.columns[-1]
        self.feature_col = support_set.columns[:-1]
        
        unique_recist_labels = support_set[self.label_col].unique()
        out_recist = set(unique_recist_labels) - set(self.prototype_class_map.keys())
        assert len(out_recist) == 0, 'Unepxected RECIST labels: %s' % out_recist
        
        prototype_features = support_set.groupby(self.label_col).mean()
        prototype_types = prototype_features.index
        prototype_representation = torch.tensor(prototype_features.values)
        prototype_representation = F.normalize(prototype_representation, p=2, dim=1)
        self.prototype_representation = prototype_representation
        self.prototype_types = prototype_types
        
        return self

    
    def transform(self, query_set):
        '''
        query_set: the last column is the RECIST label column
        '''
        label_col = query_set.columns[-1]
        assert label_col == self.label_col, '%s is missing!' % self.label_col

        query_set_features = torch.tensor(query_set[self.feature_col].values)
        query_set_features = F.normalize(query_set_features, p=2, dim=1)
        
        similarities = torch.mm(query_set_features, self.prototype_representation.T) # X*W + B
        probabilities = F.softmax(similarities / self.temperature, dim=1)
        
        probabilities = probabilities.detach().numpy()
        
        dfprob = pd.DataFrame(probabilities, index = query_set.index, columns = self.prototype_types)
        dfpred = dfprob.idxmax(axis=1).map(self.prototype_class_map)
        dftrue = query_set[self.label_col].map(self.prototype_class_map)
        
        dfprob2 = dfprob.copy()
        dfprob2.columns = dfprob2.columns.map(self.prototype_class_map)
        dfprob2 = dfprob2.T.reset_index().groupby(self.label_col).sum().T
        return dfprob2



class RegDecoder(nn.Module):
    def __init__(self, input_dim=32, dense_layers = [], out_dim = 1, dropout_p = 0.0, batch_norms = True, seed = 42):
        '''
        Regression
        '''
        super(RegDecoder,self).__init__()
        self.seed = seed
        fixseed(seed = seed)

        ## Input
        self.input_norm = torch.nn.BatchNorm1d(input_dim)

        _dense_layers = [input_dim]
        _dense_layers.extend(dense_layers)
        self._dense_layers = _dense_layers
        self.batch_norms = batch_norms
        
        ## Dense
        self.lins = ModuleList()
        for i in range(len(_dense_layers)-1):
            lin = Linear(_dense_layers[i], _dense_layers[i+1])
            self.lins.append(lin)

        ## Batchnorm
        self._batch_norms = ModuleList()
        for i in range(len(_dense_layers)-1):
            self._batch_norms.append(deepcopy(torch.nn.BatchNorm1d(_dense_layers[i+1])))

        ## Dropout
        self.dropout = nn.Dropout(dropout_p)

        # Output layer
        last_hidden = _dense_layers[-1]
        self.out = Linear(last_hidden, out_dim)


    def forward(self, x):
        if self.batch_norms & (len(self._batch_norms) == 0):
            x = self.input_norm(x)
            
        for lin, norm in zip(self.lins, self._batch_norms):
            if self.batch_norms:
                x = self.dropout(F.relu(norm(lin(x)), inplace=False))
            else:
                x = self.dropout(F.relu(lin(x), inplace=False))                
        y = self.out(x)
        return y

