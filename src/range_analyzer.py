import sys
import numpy as np
import re
import csv
from gurobipy import *
import time

from helper_functions import *

def analyze(nn, LB_N0, UB_N0, label):  
    numlayer = nn.numlayer 

    #setting up variables for propagation (dimensions will change)
    input_nodes = []
    for i in range(len(LB_N0)):
        input_nodes.append((LB_N0[i], UB_N0[i]))
    output_nodes = []

    for layer_n_ in range(numlayer):
        if(nn.layertypes[layer_n_] in ['ReLU', 'Affine']):
            weights = nn.weights[layer_n_]
            biases = nn.biases[layer_n_]

            #affine
            for n_node in range(len(weights)):
                node_value = (biases[n_node], biases[n_node])
                for j in range(len(weights[n_node])):
                    opt1 = input_nodes[j][0] * weights[n_node][j]
                    opt2 = input_nodes[j][1] * weights[n_node][j]
                    node_value = (
                        node_value[0] + min(opt1, opt2),
                        node_value[1] + max(opt1, opt2))
                output_nodes.append(node_value)

            # handle ReLU layer
            if(nn.layertypes[layer_n_]=='ReLU'):
                for n_node in range(len(output_nodes)):
                    if output_nodes[n_node][1] < 0.0:
                        output_nodes[n_node] = (0.0, 0.0)
                    elif output_nodes[n_node][0] < 0.0 and output_nodes[n_node][1] > 0.0:
                        output_nodes[n_node] = (0.0, output_nodes[n_node][1])

            #assigning next IO
            input_nodes = output_nodes
            output_nodes = []

    final_nodes = input_nodes

    # if epsilon is zero, try to classify else verify robustness 
    verified_flag = True
    predicted_label = 0
    #classification, does not change verified_flag
    if(LB_N0[0]==UB_N0[0]): # LB_N0[0]==UB_N0[0]
        for i in range(len(final_nodes)):
            inf = final_nodes[i][0]
            flag = True
            for j in range(len(final_nodes)):
                if(j!=i):
                   sup = final_nodes[j][1]
                   if(inf<=sup):
                      flag = False
                      break
            if(flag):
                predicted_label = i
                break    
    else:
        print(final_nodes)
        inf = final_nodes[label][0]
        for j in range(len(final_nodes)):
            if(j!=label):
                sup = final_nodes[j][1]
                if(inf<=sup):
                    predicted_label = label
                    verified_flag = False
                    break
     
    return predicted_label, verified_flag



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
    
    label, _ = analyze(nn,LB_N0,UB_N0,0)
    start = time.time()
    if(label==int(x0_low[0])):
        LB_N0, UB_N0 = get_perturbed_image(x0_low,epsilon)
        _, verified_flag = analyze(nn,LB_N0,UB_N0,label)
        if(verified_flag):
            print("verified")
        else:
            print("can not be verified")  
    else:
        print("image not correctly classified by the network. expected label ",int(x0_low[0]), " classified label: ", label)
    end = time.time()
    print("analysis time: ", (end-start), " seconds")
    

