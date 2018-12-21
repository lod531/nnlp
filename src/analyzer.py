import sys
import numpy as np
import re
import csv
from gurobipy import *
import time
from functools import reduce
from pprint import pprint
import math
import copy

from helper_functions import *
import range_functions as rng
import lp_functions as lp
import verification as verification

#python3 analyzer.py ../mnist_nets/mnist_relu_3_10.txt ../mnist_images/img1.txt 0.01
#python3 analyzer.py ../mnist_nets/mnist_relu_9_100.txt ../mnist_images/img1.txt 0.005
#python3 analyzer.py ../mnist_nets/mnist_relu_6_20.txt ../mnist_images/img1.txt 0.021

DEBUG = False

np.set_printoptions(suppress=True)

def verify(nn, complete_node_relus, label, time_budgets, lp_bounds = None, box_relus = None, early_stopping = True):
    n_layers = nn.numlayer
    verified_flag = False

    if lp_bounds is None:
        #need to create a copy of the relus, so that we can compare multiple versions. 
        box_relus = copy.deepcopy(complete_node_relus)
        lp_bounds = copy.deepcopy(complete_node_relus)

        #empty lp_bounds
        for i in range(1, len(lp_bounds)):
            for j in range(len(lp_bounds[i])):
                lp_bounds[i][j] = (np.nan, np.nan)

    _, _, verified_flag = compute_LP_slice(nn, lp_bounds, box_relus, nn.numlayer, nn.numlayer, label, time_budgets, run_LP_verification=True,
            full_constraints=True, early_stopping = early_stopping)

    if DEBUG: 
        if verified_flag:
            print("verified by DEV: True")
        else:
            print("verified by DEV: False")

    return None, verified_flag

def compute_LP_slice(nn, lp_bounds, relu_bounds, layer, depth, label, time_budgets, full_constraints=True, run_LP_verification=False, early_stopping = False):
    m = lp.create_LP_Model()
    pseudo_input_layer = rng.fix_relu_bounds(lp_bounds[layer - depth])
    linear_net = lp.initialize_lin_net_with_nodes(m, pseudo_input_layer)

    lp_bounds_slice = lp_bounds.copy()
    relu_bounds_slice = relu_bounds.copy()
    
    m, linear_net, lp_bounds, box_relus, verified_flag = lp.create_lin_net(m, time_budgets, linear_net, layer-depth, depth, nn,
        lp_bounds = lp_bounds_slice, box_relus = relu_bounds_slice, label=label, early_stopping = early_stopping)

    if not verified_flag:
        if run_LP_verification and  ( (not verified_flag) or DEBUG):
            _, lp_verification = verification.class_difference(m, linear_net, label)
            if lp_verification:
                verified_flag = True

    return lp_bounds, box_relus, verified_flag

def get_budgets(nn, label, total_budget=410):

    verification_budget = 90
    if len(nn.weights[0]) <= 400:
        verification_budget = 30
    elif label != 1:
        verification_budget = 60

    num_relu_layers = 0
    for ind, layer_type in enumerate(nn.layertypes):
        if layer_type == 'ReLU' and ind < len(nn.layertypes) - 1:
            num_relu_layers += 1

    budget_per_layer = (total_budget - verification_budget) / num_relu_layers

    time_budgets = []
    for ind, layer_type in enumerate(nn.layertypes):        
        if layer_type == 'ReLU' and ind < len(nn.layertypes) - 1:
            time_budgets.append(budget_per_layer)
        else:
            time_budgets.append(0.0)

    if DEBUG: print(time_budgets)
    return time_budgets

def analyze(nn, LB_N0, UB_N0, label, epsilon):  
    timing = []
    verified_flag = False

    if DEBUG: print("----------------------\nBounds")
    ttmp = time.time()
    node_relus = rng.update_relus_with_new_bounds(nn, [[]], rng.create_input_tuples(LB_N0, UB_N0),0)

    #trying to verify the network using ranges
    predicted_label, verified_flag = rng.verify_bounds(node_relus[-1], label)
    timing.append(('bounds', time.time()-ttmp))
    
    if DEBUG:
        print("range bounds:\n" + str(node_relus[-1]))
        print("verified by bounds: " + str(verified_flag) + "\n")

    #if verified, return
    if verified_flag and not DEBUG:
        return predicted_label, verified_flag

    if DEBUG: print("----------------------\nFull LP")

    time_budgets = get_budgets(nn, label)
    ttmp = time.time()
    _, verified_flag = verify(nn, node_relus, label, time_budgets)
    timing.append(('full LP', time.time()-ttmp))

    if DEBUG:
        print('\ntiming results: ' + str(timing))

    return None, verified_flag


if __name__ == '__main__':
    from sys import argv
    if len(argv) < 3 or len(argv) > 4:
        print('usage: python3.6 ' + argv[0] + ' net.txt spec.txt [timeout]')
        exit(1)
    netname = argv[1]
    specname = argv[2]
    epsilon = float(argv[3])
    #c_label = int(argv[4])
    with open(netname, 'r') as netfile:
        netstring = netfile.read()
    with open(specname, 'r') as specfile:
        specstring = specfile.read()
    nn = parse_net(netstring)
    x0_low, x0_high = parse_spec(specstring)
    LB_N0, UB_N0 = get_perturbed_image(x0_low,0)
    label = rng.classify_for_zero_epsilon(nn,LB_N0,UB_N0)
    start = time.time()
    if(True or label==int(x0_low[0])):
        LB_N0, UB_N0 = get_perturbed_image(x0_low,epsilon)
        _, verified_flag = analyze(nn,LB_N0,UB_N0,label,epsilon)
        if(verified_flag):
            print("verified")
        else:
            print("can not be verified")  
    else:
        print("image not correctly classified by the network. expected label ",int(x0_low[0]), " classified label: ", label)
    end = time.time()
    print("analysis time: ", (end-start), " seconds")

#'''
#print("----------------------\nDynamic depth selection")
#ttmp = time.time()
#verified_flag = dynamic_depth_verification(nn, node_relus, label, start_time)
#timing.append(('Dynamic depth selection', time.time()-ttmp))
##'''
#def dynamic_depth_verification(nn, complete_node_relus, label, start_time, full_constraints=True):
#    box_relus = copy.deepcopy(complete_node_relus)
#    lp_bounds = copy.deepcopy(complete_node_relus)
#    #empty lp_bounds
#    for i in range(1, len(lp_bounds)):
#        for j in range(len(lp_bounds[i])):
#            lp_bounds[i][j] = (np.nan, np.nan)
#    m = lp.create_LP_Model()
#    pseudo_input_layer = rng.fix_relu_bounds(lp_bounds[0])
#    linear_net = lp.initialize_lin_net_with_nodes(m, pseudo_input_layer)
#    m, linear_net, lp_bounds, box_relus, verified_flag, depth = lp.find_dynamic_depth(m, linear_net, 0, nn.numlayer, nn, lp_bounds = lp_bounds, 
#                                                                                box_relus = box_relus, label = label, start_time = start_time)
#    print('vflag', verified_flag)
#    print('depth', depth)
#
#    if not verified_flag:
#        if depth == nn.numlayer:
#            _, lp_verification = verify.class_difference(m, linear_net, label)
#            if lp_verification:
#                verified_flag = True
#
#        if not verified_flag:
#            instructions = []
#            for i in range(depth+1, nn.numlayer + 1):
#                instructions.append({'layer':i, 'depth':depth, 'full_constraints':True})
#            _, verified_flag = verify(nn, box_relus, instructions, label, lp_bounds, box_relus)
#        else:
#            return verified_flag
#    else:
#        return verified_flag
