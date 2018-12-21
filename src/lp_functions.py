import sys
import time
import numpy as np
import re
import csv
from gurobipy import *
from functools import reduce
from pprint import pprint
import copy

import range_functions as rng

DEBUG = False

def create_lp_relu(m, lb, ub, linear_affine, nn_layer, neuron):
    if(ub <= 0):
        #create variable whic his 0
        relu = m.addVar(lb = 0, ub = 0, vtype='C', 
                name = 'ReLU layer ' + str(nn_layer) + ' neuron ' + str(neuron)) 
    elif (lb >= 0): 
        #create variable which is equal to the linear affine
        relu = m.addVar(lb = lb, ub = ub, vtype='C', 
                name = 'ReLU layer ' + str(nn_layer) + ' neuron ' + str(neuron))
        m.addConstr(relu == linear_affine)
    else: # ub > 0 and lb < 0
        #relu must be >0, >= affine and <= the new diagonal line (linear_relu_approximation)
        relu = m.addVar(lb = 0, ub = ub, vtype='C')
#                name = 'ReLU layer ' + str(nn_layer) + ' neuron ' + str(neuron))
        slope = ub/(ub - lb)
        intercept = slope * (-lb)
        linear_relu_approximation = linear_affine * slope + intercept
        
        #ReLU is less than the approximation at each point
        m.addConstr(relu <= linear_relu_approximation)
        m.addConstr(relu >= linear_affine)
    return relu

def create_affine_node(m, lb, ub, linear_affine, nn_layer, neuron):
    node = m.addVar(lb = lb, ub = ub, vtype='C', 
                name = 'Affine layer ' + str(nn_layer) + ' neuron ' + str(neuron))
    m.addConstr(node == linear_affine)
    return node

def create_LP_Model():
    m = Model("Full LP")
    m.setParam( 'OutputFlag', False )
    return m

def initialize_lin_net_LB_UB(m, LB_N0, UB_N0):
    linear_net = [[]]

    # add input as variables with lb, ub to the model, and the variable to the linear expression matrix
    for i in range(0, len(LB_N0)):
        linear_net[0].append(m.addVar(lb = LB_N0[i], ub = UB_N0[i], vtype='C',
                                        name = str(i)))
    return linear_net

def initialize_lin_net_with_nodes(m, nodes):
    linear_net = [[]]
    for i in range(0, len(nodes)):
        linear_net[0].append(m.addVar(lb = nodes[i][0], ub = nodes[i][1], vtype='C',
                                        name = str(i)))
    return linear_net

def compute_LP_bounds(m, linear_affine, is_affine_layer=False):
    m.setObjective(linear_affine, GRB.MAXIMIZE)
    m.optimize()
    ub = m.getObjective().getValue()
    if not is_affine_layer and ub <= 0.0:
        return 0.0, 0.0
    m.setObjective(linear_affine, GRB.MINIMIZE)
    m.optimize()
    lb = m.getObjective().getValue()
    if(ub < lb):
        print('lol')
        print('ub', ub)
        print('lb', lb)
    assert(ub >= lb)
    return lb, ub

def compute_node_bounds_with_LP(m, linear_net):
    last_layer = linear_net[-1]
    bounds = []
    for i in range(len(last_layer)):
        lb, ub = compute_LP_bounds(m, last_layer[i])
        bounds.append((lb, ub))
    return bounds

def LP_given_layer_neurons(m, time_budget, box_relus, box_bounds, lp_bounds, weights, biases, layertypes, 
        nn_layer, lp_layer, lin_net_layer, starting_layer, linear_net, neurons, nn, label):

    is_affine_layer = layertypes[nn_layer] == 'Affine'
    linear_affines = [None for i in range(len(weights[nn_layer]))]

    start = time.time()
    tmp_time_neuron = []

    for ind, neuron in enumerate(neurons):
        linear_affine = LinExpr(biases[nn_layer][neuron])
        linear_affine.addTerms(weights[nn_layer][neuron], linear_net[lin_net_layer-1])
        if lp_bounds is not None and (not np.isnan(lp_bounds[lp_layer][neuron][0])) and (not np.isnan(lp_bounds[lp_layer][neuron][1])):
            lb = lp_bounds[lp_layer][neuron][0]
            ub = lp_bounds[lp_layer][neuron][1]
        elif box_relus is not None and box_relus[lp_layer][neuron][1] <= 0.0 and not is_affine_layer:
            lb = ub = 0.0
        elif is_affine_layer or (nn_layer == starting_layer and box_bounds is not None) or \
                ((len(tmp_time_neuron) > 0) and (time.time() - start + (np.array(tmp_time_neuron).mean())) >= time_budget):
            lb, ub = box_bounds[neuron]
        else:
            tmp_time = time.time()
            lb, ub = compute_LP_bounds(m, linear_affine, is_affine_layer)
            tmp_time_neuron.append(time.time() - tmp_time)

        lp_bounds[lp_layer][neuron] = (lb, ub)  
        linear_affines[neuron] = linear_affine

    #print("ratio of layer computed with LP bounds: " + str(len(tmp_time_neuron)) + "/" + str(len(neurons)))

    for neuron in neurons:
        lb, ub = lp_bounds[lp_layer][neuron]
        if layertypes[nn_layer] == 'Affine':
            node = create_affine_node(m, lb, ub, linear_affines[neuron], nn_layer, neuron)
        else: #must be RELU:
            node = create_lp_relu(m, lb, ub, linear_affines[neuron], nn_layer, neuron)
        linear_net[lin_net_layer][neuron] = node

    return False, time.time()-start


def create_lin_net(m, time_budgets, linear_net, starting_layer, n_layers, nn, lp_bounds = None, box_relus = None, 
                    label = None, early_stopping=False):
    verified_label = False

    weights = nn.weights
    biases = nn.biases
    layertypes = nn.layertypes
    
    #creating 'infinite' time-budgets if None (24h total, excluding verification). 
    if time_budgets is None:
        time_budgets = [(86400/nn_layers) for i in range(n_layers)]

    for nn_layer in range(starting_layer, starting_layer+n_layers):
        start = time.time()
        lp_layer = nn_layer + 1
        lin_net_layer = (nn_layer - starting_layer) + 1

        #setup
        linear_net.append([None for i in range(len(weights[nn_layer]))])
        box_bounds = rng.get_box_bounds(nn, box_relus, nn_layer)        
        sorted_neuron_indices = rng.problem_children(box_relus[nn_layer+1], (weights[nn_layer+1] if len(weights) > nn_layer + 1 else None))[:,1].astype(int)

        #update the LP for current layer
        verified_flag, time_taken = LP_given_layer_neurons(
            m, time_budgets[nn_layer], 
            box_relus, box_bounds, lp_bounds, weights, biases, layertypes, 
            nn_layer, lp_layer, lin_net_layer, starting_layer, 
            linear_net, sorted_neuron_indices, nn, label)

        #udpating the budgets
        if nn_layer < nn.numlayer - 1:
            time_budgets[nn_layer+1] = max(0, time_budgets[nn_layer+1] + (time_budgets[nn_layer] - time_taken))
        time_budgets[nn_layer] = time_taken
        
        #propagating bounds, and attempting verification
        box_relus = rng.update_relus_with_new_bounds(nn, box_relus, rng.combine_partial_lp_with_relu(lp_bounds[lp_layer], box_relus[lp_layer]), lp_layer)
        _, verified_flag = rng.verify_bounds(box_relus[-1], label)

        #debug prints
        if DEBUG:
            print('LP formulation layer ' + str(nn_layer+1-starting_layer) + "/" + str(n_layers) + " completed in " + str(time.time() - start) + "s")
            if verified_flag:
                print("    verified by box-propagation")
            print("updated time budgets: " + str(time_budgets))

        #early stopping
        if verified_flag and early_stopping:
            return m, linear_net, lp_bounds, box_relus, verified_flag

    return m, linear_net, lp_bounds, box_relus, verified_label

'''
def deprecated_LP_given_layer_neurons(m, box_relus, box_bounds, lp_bounds, weights, biases, layertypes, nn_layer, lp_layer, 
                            lin_net_layer, linear_net, neurons, starting_layer, linear_affines, dynamic_constraints, start_time, nn, label, early_stopping):
    is_affine_layer = layertypes[nn_layer] == 'Affine'

    update_count = 0
    for neuron in neurons:
        linear_affine = LinExpr(biases[nn_layer][neuron])
        linear_affine.addTerms(weights[nn_layer][neuron], linear_net[lin_net_layer-1])
            
        if lp_bounds is not None and (not np.isnan(lp_bounds[lp_layer][neuron][0])) and (not np.isnan(lp_bounds[lp_layer][neuron][1])):
            lb = lp_bounds[lp_layer][neuron][0]
            ub = lp_bounds[lp_layer][neuron][1]
        elif box_relus is not None and box_relus[lp_layer][neuron][1] <= 0.0 and not is_affine_layer:
            lb = ub = 0.0
        elif nn_layer == starting_layer and box_bounds is not None:
            lb, ub = box_bounds[neuron]
        else:
            lb, ub = compute_LP_bounds(m, linear_affine, is_affine_layer)
            update_count += 1
            if early_stopping and update_count == 5 and nn_layer > 1:
                update_count = 0
                box_relus = rng.update_relus_with_new_bounds(nn, box_relus, rng.combine_partial_lp_with_relu(lp_bounds[lp_layer], box_relus[lp_layer]), lp_layer)
                _, verified_flag = rng.verify_bounds(box_relus[-1], label)
                if verified_flag:
                    print("Solved at neuron", neuron, "which is at index", np.where(neurons == neuron))
                if verified_flag and early_stopping:
                    return True
        lp_bounds[lp_layer][neuron] = (lb, ub)
            
        linear_affines[neuron] = linear_affine

    for neuron in neurons:
        lb, ub = lp_bounds[lp_layer][neuron]
        if layertypes[nn_layer] == 'Affine':
            node = create_affine_node(m, lb, ub, linear_affines[neuron], nn_layer, neuron)
        else: #must be RELU:
            node = create_lp_relu(m, lb, ub, linear_affines[neuron], nn_layer, neuron)
        linear_net[lin_net_layer][neuron] = node

    return False


def deprecated_create_lin_net(m, tmp_budgets, linear_net, starting_layer, n_layers, nn, lp_bounds = None, box_relus = None, 
                    label = None, dynamic_constraints=True, early_stopping=False, start_time = None):
    verified_label = False
    box_bounds = None

    weights = nn.weights
    biases = nn.biases
    layertypes = nn.layertypes
    previous_runtime = 0

    for nn_layer in range(starting_layer, starting_layer+n_layers):
        start = time.time()
        lp_layer = nn_layer + 1
        lin_net_layer = (nn_layer - starting_layer) + 1

        linear_net.append([None for i in range(len(weights[nn_layer]))])
        linear_affines = [None for i in range(len(weights[nn_layer]))]
        if nn_layer == starting_layer:
            box_bounds = rng.get_box_bounds(nn, box_relus, nn_layer)
        
        if early_stopping and nn_layer != n_layers-1:
            #[:,1] to just get the index column
            neurons = rng.problem_children(box_relus[nn_layer+1], weights[nn_layer+1])[:,1].astype(int)
        else:
            neurons = np.array([i for i in range(len(nn.weights[nn_layer]))])
        verified_flag = LP_given_layer_neurons(m, box_relus, box_bounds, lp_bounds, weights, biases, layertypes, 
                                nn_layer, lp_layer, lin_net_layer, linear_net, neurons, starting_layer, linear_affines, dynamic_constraints, start_time, nn, label, early_stopping)
        box_relus = rng.update_relus_with_new_bounds(nn, box_relus, rng.combine_partial_lp_with_relu(lp_bounds[lp_layer], box_relus[lp_layer]), lp_layer)

        if verified_flag and early_stopping:
            return m, linear_net, lp_bounds, box_relus, verified_flag

        if DEBUG:
            runtime = time.time() - start
            print('LP formulation layer ' + str(nn_layer+1-starting_layer) + "/" + str(n_layers) + " completed in " + str(runtime) + "s")
            #if previous_runtime != 0:
            #    print('Ratio between last and this neuron runtimes:', (runtime/len(nn.weights[nn_layer]))/(previous_runtime/len(nn.weights[nn_layer-1])))
            previous_runtime = runtime

    return m, linear_net, lp_bounds, box_relus, verified_label
'''