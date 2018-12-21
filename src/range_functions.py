import sys
import numpy as np
import re
import csv
from gurobipy import *
from functools import reduce
from pprint import pprint

def create_input_tuples(LB_N0, UB_N0):
    return np.column_stack((LB_N0, UB_N0))

def create_range_bounds(nn, input_bounds):
    return update_relus_with_new_bounds(nn, [[]], input_bounds, 0)

def update_relus_with_new_bounds(nn, node_relus, layer_relus, given_layer):
    n_layers = nn.numlayer
    node_relus = node_relus[0:given_layer]
    node_relus.append(layer_relus)
    for layer_n in range(given_layer, n_layers):
        #layers going forward
        combined_node_bounds = get_box_bounds(nn, node_relus, layer_n)
        if(nn.layertypes[layer_n]=='ReLU'):
            combined_relu_bounds = get_relu_bounds(combined_node_bounds)
            node_relus.append(combined_relu_bounds)
        else: #affine
            node_relus.append(combined_node_bounds)
    return node_relus

def get_relu_bounds(combined_node_bounds):
    #for the relus, I want, for the lower bound the max(0, lower_bound), which is fine.
    #and for the upper bound I want the max(0, upper_bound). Note that if it turns
    #out that upper_bound <= 0, then lower_bound <= 0 and so the whole thing is (0, 0)
    lower_bound = combined_node_bounds[:,0]
    upper_bound = combined_node_bounds[:,1]
    zeroes = np.zeros((len(lower_bound)))
    lower_bound_relu = np.where(zeroes >= lower_bound, zeroes, lower_bound)
    upper_bound_relu = np.where(zeroes >= upper_bound, zeroes, upper_bound)
    combined_relu_bounds = np.column_stack((lower_bound_relu, upper_bound_relu))
    return combined_relu_bounds

def get_box_bounds(nn, current_bounds, given_layer):
    '''
    current bounds is a list of bounds which must contain at least given_layer
    number of layers.
    given layer is the index of the layer upon which the box bounds will be based on
    returns box bounds based on, well, given_layer
    '''
    weights = nn.weights[given_layer]
    biases = nn.biases[given_layer]
    reshaped_relus = np.transpose(current_bounds[given_layer])
    lower_bounds = reshaped_relus[0]
    lower_bounds_matrix = np.repeat([lower_bounds], len(weights), axis=0)
    upper_bounds = reshaped_relus[1]
    upper_bounds_matrix = np.repeat([upper_bounds], len(weights), axis=0)

    lower_bounds_matrix = lower_bounds_matrix * weights
    upper_bounds_matrix = upper_bounds_matrix * weights

    affine_components_min = np.where(lower_bounds_matrix <= upper_bounds_matrix, lower_bounds_matrix, upper_bounds_matrix)                                                                      
    affine_components_max = np.where(lower_bounds_matrix >= upper_bounds_matrix, lower_bounds_matrix, upper_bounds_matrix)                                                                      

    lower_bound = np.sum(affine_components_min, axis=1) + biases
    upper_bound = np.sum(affine_components_max, axis=1) + biases
    return np.column_stack((lower_bound, upper_bound))

def classify_for_zero_epsilon(nn, LB_N0, UB_N0):
    n_layers = nn.numlayer
    relu_bounds = create_range_bounds(nn, create_input_tuples(LB_N0, UB_N0))
    output_relu = relu_bounds[n_layers]
    predicted_label = 0

    if(LB_N0[0]==UB_N0[0]):
        current_max = output_relu[0][0]
        predicted_label = 0
        for i in range(0, len(output_relu)):
            if current_max < output_relu[i][0]:
                predicted_label = i
                current_max = output_relu[i][0]
        return predicted_label
    else:
        return None

def verify_bounds(output_bounds, label):
    verified_flag = True
    predicted_label = 0
    inf = output_bounds[label][0]
    for j in range(len(output_bounds)):
        if(j!=label):
            sup = output_bounds[j][1]
            if(inf<=sup):
                predicted_label = label
                verified_flag = False
                break
    return predicted_label, verified_flag

def fix_relu_bounds(bounds):
    return get_relu_bounds(bounds)


def problem_children(ranges, weights=None):
    '''
    ranges is a wd matrix of two columns, col 0 = lower bounds, col 1 = upper bounds
    weights is a 2d matrix holding the weights for the layer in question
    '''
    differences = ranges[:,1] - ranges[:,0] 
    if weights is not None:   
        for row in range(weights.shape[1]):
            differences[row] = np.sum(np.abs(differences[row] * weights[:,row]))
    #append indexes on to differences
    #indexes is a column of indexes.
    indexes = np.arange(0, len(differences)).reshape(len(differences), 1)
    #append indexes on to differences
    differences = np.column_stack((differences, indexes))
    #Don't even try to understand this. Google 'numpy sort by column'
    differences = differences[differences[:,0].argsort()[::-1]]
    return differences


def combine_partial_lp_with_relu(partial_lp_layer, box_relus):
    combined_layer = np.where(np.logical_not(np.isnan(partial_lp_layer)), partial_lp_layer, box_relus)
    relu = get_relu_bounds(combined_layer)
    return relu
