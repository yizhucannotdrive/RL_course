import numpy as np
from matplotlib import pyplot as plt

K = 15
cf = 100
ch =2
num_limit = 200
period  = 500
gamma = 0.95
# a takes 0 or 1, 0 means not dispatch while a=1 mean dispatch
arrival_p = dict()
value_iteration_rep_max = 1000
K_policy = 10
epsilon = 1e-3
for i in range(1,6):
    arrival_p[i] = 0.2

def reward(s,a):
    if a == 0:
        return - ch * s
    else:
        return -(cf+ch * max([0,s-K]))
def next_state(s,arrival, a):
    if a == 0:
        return min([s + arrival, num_limit])
    else:
        return min([max([s-K,0])+arrival, num_limit])
# arrival_p is a dict with keys as r.vs support
def cal_expectation_V(V,s,a, arrival_p):
    exp_v = 0
    for key in arrival_p.keys():
        exp_v += arrival_p[key] * V[next_state(s,key,a)]
    return exp_v

def main():
    # do enumerate
    states = range(0,num_limit+1)
    V_t = dict()
    a_t = dict()
    V_tplus1 = dict()
    #initialization
    for s in states:
        V_tplus1[s] = 0
    t = period
    ####################################################################################
    print("Enumerate case")
    while t>=0:
        for s in states:
            v1 = reward(s,1) + gamma * cal_expectation_V(V_tplus1,s,1,arrival_p)
            v0 = reward(s,0) + gamma * cal_expectation_V(V_tplus1,s,0,arrival_p)
            V_t[s] = max([v1, v0])
            if v1>v0:
                a_t[s] = 1
            else:
                a_t[s] = 0
        V_tplus1 = V_t
        t = t-1
    print("optimal strategy is {}".format(a_t))
    ss = V_t.keys()
    values = [-i for i in V_t.values()]
    plt.plot(ss, values, 'ro')
    plt.xlabel('s')
    plt.ylabel('V(s)')
    plt.title("Enumeration")
    plt.show()
    ##################################################################################3
    print("Value iteration case")
    V = dict()
    a = dict()
    for s in states:
        V[s] = 0
    V_vals = V.values()
    num_iter = 0
    while num_iter < value_iteration_rep_max:
        V_pre_vals = V_vals
        for s in states:
            v1 = reward(s,1) + gamma * cal_expectation_V(V,s,1,arrival_p)
            v0 = reward(s,0) + gamma * cal_expectation_V(V,s,0,arrival_p)
            V[s] = max([v1, v0])
            if v1>v0:
                a[s] = 1
            else:
                a[s] = 0
        V_vals = V.values()
        num_iter +=1
        if np.max(np.abs(np.array(V_vals)-np.array(V_pre_vals)))<epsilon:
            break
    print("optimal strategy is {}".format(a))
    ss = V.keys()
    values = [-i for i in V.values()]
    plt.plot(ss, values, 'bo')
    plt.xlabel('s')
    plt.ylabel('V(s)')
    plt.title("Value Iteration")
    plt.show()

    ######################################################################################
    print("Policy iteration case")
    V = dict()
    a = dict()
    for s in states:
        V[s] = 0
        a[s] = 0
    num_iter = 0
    while num_iter < value_iteration_rep_max:
        num_iter_inner = 0
        V_vals = V.values()
        V_vals_inner = V_vals
        while num_iter_inner < value_iteration_rep_max:
            V_pre_vals = V_vals_inner
            for s in states:
                V[s] = reward(s, a[s]) + gamma * cal_expectation_V(V, s, a[s], arrival_p)
            V_vals_inner = V.values()
            num_iter_inner += 1
            if np.max(np.abs(np.array(V_vals_inner) - np.array(V_pre_vals))) < epsilon:
                break
        for s in states:
            v1 = reward(s,1) + gamma * cal_expectation_V(V,s,1,arrival_p)
            v0 = reward(s,0) + gamma * cal_expectation_V(V,s,0,arrival_p)
            if v1>v0:
                a[s] = 1
            else:
                a[s] = 0
        num_iter += 1
        if np.max(np.abs(np.array(V_vals_inner) - np.array(V_vals))) < epsilon:
            break
    print("optimal strategy is {}".format(a))
    ss = V.keys()
    values = [-i for i in V.values()]
    plt.plot(ss, values, 'go')
    plt.xlabel('s')
    plt.ylabel('V(s)')
    plt.title("Policy Iteration")
    plt.show()


if __name__ == "__main__":
    main()
