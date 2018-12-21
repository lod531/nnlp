from gurobipy import *

m = Model('main_model')

a = m.addVar(lb = - GRB.INFINITY, ub = 4, vtype = 'C', name = 'a')
b = m.addVar(lb = -2, ub = 1, vtype = 'C', name = 'b')
c = m.addVar(lb = -4, ub = 3, vtype = 'C', name = 'c')
d = m.addVar(lb = -3, ub = 10, vtype = 'C', name = 'd')
e = m.addVar(lb = -3, ub = 3, vtype = 'C', name = 'e')
f = m.addVar(lb = -3, ub = 3, vtype = 'C', name = 'f')
#g = m.addVar(lb = -3, ub = 3, vtype = 'C')
#h = m.addVar(lb = -3, ub = 3, vtype = 'C')
lin1 = a + b + c
lin2 = - a - b - c
lin3 = LinExpr(-2)

#m.addConstr(lin1 >= 0)
m.setObjective(d, GRB.MAXIMIZE)
m.addConstr(d == lin1)
print('OBJ', m.getObjective())

m.optimize()
print(a)

print('TEST', m.getObjective().getValue())

for v in m.getVars():
    print(v.varName, v.x)
print('okay')
