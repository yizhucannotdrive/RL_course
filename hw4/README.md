It's an implementation of UCB contexutal bandit algorithm -- LinUCB with disjoint linear models
http://rob.schapire.net/papers/www10.pdf
Basically, I used linear models for each arm and update the selected arm using context as feature vector.
The pipeline can work, but result seems it doens't learn much.
![alt text](https://github.com/yizhucannotdrive/RL_course/blob/master/hw4/regret.png)
