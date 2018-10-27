# RL_course
HWs of RL course northwestern
It's a repository for hws in RL course in Northwestern given by prof Klanjan.

HW1 is a standard exercise of bus/flight dispatch problem. 

1) Consider shuttle dispatching problem in which a dispatcher is responsible to decide whether or not
to dispatch a shuttle at each decision epoch depending on number of customers waiting for the service. A
standard shuttle dispatch problem has following parameters:
K = The capacity of a shuttle if it is dispatched.
A t = A random variable giving the number of customers arriving during time interval t.
c f = The cost of dispatching a shuttle.
c h = The cost per customer left waiting per time period.
Given K = 15, c f = 100, c h = 2 and assuming that A t follows unif{1,5}, model the problem and solve it
using:
a) Enumeration (with time period T =500)
b) Value iteration (T = ∞)
c) Policy iteration.
You can assume that number of people in station can not exceed 200 and discount rate γ = 0.95.
For part (a), plot optimal value function at time 0 versus number of customers waiting.
For part (b), plot optimal value function versus number of customers waiting.
For part (c), plot optimal policy versus number of customers waiting.
2) Now consider the multiclass problem in which we have different types of customers. Assume that there
are 5 types of customers with c h = {1, 1.5, 2, 2.5, 3} and each type can have maximum 100 people of each
class waiting for shuttle and A t for each class follows same distribution. Capacity of the shuttle is K = 30.
Try to repeat a), b) and c) from problem 1.
![alt text](http://url/to/img.png)
