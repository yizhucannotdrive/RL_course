It's a multi-thread  dqn implementation for game Breakout-v0 and MsPacman game.
In detail, the implementation includes replay buffer and target network. The sampling scheme is designed as epsilon-greedy where epsilon decays at a certain rate.
The result of Breakout-v0 is not satisfying, though there is some trend of reward increasing.
The graph below is the performance of Breakout-v0

![alt text](https://github.com/yizhucannotdrive/RL_course/blob/master/hw3/breakout.png)
It contains the loss function, q value estimation and return for each episode. Unfortunately, the return grows at beginning but stop increasing at some point. It seems it might get stuck at some local minimum.

The pipeline should work, since it works for CartPole and MsPacman. The bad result might be due to the  sensitivity of choice of network

The result of MsPacman is good.

The graph below is the peformance of MsPacman0-v0.
![alt text](https://github.com/yizhucannotdrive/RL_course/blob/master/hw3/MsPacman1.png)
![alt text](https://github.com/yizhucannotdrive/RL_course/blob/master/hw3/MsPacman2.png)
