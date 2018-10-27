import numpy as np
from matplotlib import pyplot as plt
import itertools

# assume most expensive customer go first
K = 30
cf = 100
ch = [1, 1.5, 2,2.5,3]
num_limit = 2
period  = 500
gamma = 0.95
# a takes 0 or 1, 0 means not dispatch while a=1 mean dispatch
arrival_p = dict()
value_iteration_rep_max = 1000
K_policy = 10
epsilon = 1e-3
first_dim_vis_rest = [1,1,1,1]
for i in range(1,6):
    arrival_p[i] = 0.2

def reward(s,a):
    s = list(s)
    if a == 0:
        return -np.dot(s,ch)
    else:
        K_rest = K
        for i in range(len(s)-1,-1, -1):
            if K_rest <= s[i]:
                s[i] = s[i]- K_rest
                K_rest = 0
                break
            else:
                K_rest -= s[i]
                s[i] = 0
        return -(cf+np.dot(s,ch))
def next_state(s,arrival, a):
    s = list(s)
    if a == 0:
        return tuple([ min([s[i] + arrival[i], num_limit]) for i in range(len(s))])
    else:
        K_rest = K
        for i in range(len(s) - 1, -1, -1):
            if K_rest <= s[i]:
                s[i] = s[i] - K_rest
                K_rest = 0
                break
            else:
                K_rest -= s[i]
                s[i] = 0
        return tuple([ min([s[i] + arrival[i], num_limit]) for i in range(len(s))])
# arrival_p is a dict with keys as r.vs support
def cal_expectation_V(V,s,a, arrival_p):
    exp_v = 0
    arrival_1dim = arrival_p.keys()
    arrivals = list(itertools.product(arrival_1dim, arrival_1dim, arrival_1dim, arrival_1dim, arrival_1dim))
    for arrival in arrivals:
            exp_v += arrival_p[arrival[0]]*arrival_p[arrival[0]]*arrival_p[arrival[0]]*arrival_p[arrival[0]]*arrival_p[arrival[0]]\
                                 * V[next_state(s,arrival,a)]
    return exp_v

def main():
    # do enumerate
    states_one_dim = list(range(0,num_limit+1))
    states=list(itertools.product(states_one_dim, states_one_dim, states_one_dim, states_one_dim, states_one_dim))
    #for i in range(len(states)):
    #    states[i] = list(states[i])
    V_t = dict()
    a_t = dict()
    V_tplus1 = dict()
    #initialization
    for s in states:
        V_tplus1[s] = 0
    t = period
    ####################################################################################
    """
    print("Enumerate case")
    while t>=0:
        print(t)
        for s in states:
            v1 = reward(s,1) + gamma * cal_expectation_V(V_tplus1,s,1,arrival_p)
            v0 = reward(s,0) + gamma * cal_expectation_V(V_tplus1,s,0,arrival_p)
            V_t[s] = max([v1, v0])
            #print(s,V_t[s])
            if v1>v0:
                a_t[s] = 1
            else:
                a_t[s] = 0
        V_tplus1 = V_t
        t = t-1
    #print("optimal strategy is {}".format(a_t))
    #print(V_t.keys())
    ss = list(states_one_dim)
    values = [-V_t[tuple([s]+first_dim_vis_rest)] for s in ss]
    plt.plot(ss, values, 'bo')
    plt.xlabel('s')
    plt.ylabel('V(s)')
    plt.title("Enumeration_multi_class")
    plt.show()
    """
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
    #print("optimal strategy is {}".format(a))
    ss = list(states_one_dim)
    values = [-V[tuple([s]+first_dim_vis_rest)] for s in ss]
    plt.plot(ss, values, 'bo')
    plt.xlabel('s')
    plt.ylabel('V(s)')
    plt.title("Value_Iterarion_multi_class")
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
    ss = list(states_one_dim)
    values = [-V[tuple([s]+first_dim_vis_rest)] for s in ss]
    plt.plot(ss, values, 'bo')
    plt.xlabel('s')
    plt.ylabel('V(s)')
    plt.title("Policy_iteration_multi_class")
    plt.show()


if __name__ == "__main__":
    main()
