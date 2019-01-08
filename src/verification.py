import sys
import numpy as np
import re
import csv
from gurobipy import *
import time
from functools import reduce
from pprint import pprint

DEBUG = False

#deprecated
def min_max(m, linear_net, label):
    m.update()
    m.setObjective(linear_net[-1][label], GRB.MINIMIZE)
    m.optimize()

    label_min = m.getObjective().getValue()
    verified_flag = True
    class_maxes = []
    for i in range(0, 10):
        m.setObjective(linear_net[-1][i], GRB.MAXIMIZE)
        m.optimize()
        class_max = m.getObjective().getValue()
        if i != label:
            class_maxes.append(class_max)
            if label_min <= class_max:
                verified_flag = False
    if DEBUG:
        print('class ' + str(label) + ', LP    min: ' + str(label_min) + '. (box: not given)')
        print('non-label class-maxes: ' + str(class_maxes))
        print("verified by min-max LP: " + str(verified_flag) + "")
    return None, verified_flag

   
def class_difference(m, linear_net, label, CURRENT_DATA):
    m.update()
    verified_flag = True
    outputs = []
    CURRENT_DATA['differences'] = {}
    start = time.time()
    for i in range(0, 10):
        if i != label:
            #print(i)
            m.setObjective(linear_net[-1][i] - linear_net[-1][label] , GRB.MAXIMIZE)
            start = time.time()
            m.optimize()
            diff = m.getObjective().getValue()
            CURRENT_DATA['differences'][i] = {}
            CURRENT_DATA['differences'][i]['diff'] = diff
            CURRENT_DATA['differences'][i]['time_taken'] = time.time() - start
            outputs.append(diff) 
            if diff > 0:
                verified_flag = False
    if DEBUG:
        print("differences: " + str(outputs))
        print("verified by class-difference LP: " + str(verified_flag) + "")
        print("time to verify: " + str(time.time() - start))

    return None, verified_flag

