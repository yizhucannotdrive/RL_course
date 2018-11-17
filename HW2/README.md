I tried to implement policy gradient on CartPole and Pong-v0.

In Carpole, it works well. After 300 episode, it gets around 200.
This the screen shot from tensorboard:

![alt text](https://github.com/yizhucannotdrive/RL_course/blob/master/HW2/cartpole.png
)

In Pong-v0, I only work out a easy version. In my code folder, file test.py is able to get slowly increasing episode reward.
The plot below plots the growing trend of first 200 episodes.


![alt text](https://github.com/yizhucannotdrive/RL_course/blob/master/HW2/200_epi.png
)

The following shows the growing trend of first 20000 episodes.

![alt text](https://github.com/yizhucannotdrive/RL_course/blob/master/HW2/20000_epi.png
)

The baseline I used in the implementation above is simply the mean of sample return. 
To workout more complicated case, I implemented a2c, you can check agent.py to see how it works. Basically, the baseline function is also a neural network.From the perspective of actor and critic, baseline function acts as a critic here. Result is still in running. 

To be more specific, here are a list of tips I implemented to make it work.

1) View state as the image difference from the previous frame

2) when calculating return, split reward sequence to 21 games and then do the return calculation iteration.

3)network is sensitive, it seems some small network even works better

4) multiple processor doesn't help much in this case, since enviroment evolves with little variance.

5)The batchsize is big that each training step is feeded with 10 episodes. If each episode has around 1000 steps, each training iteration has a batch size of around 10000.

6) I added a NOOP operation to stablize the action space

7)When recording the episode reward, it's important to record moving average rather than single episode mean reward since noise might make recognizing trend very difficult.

Sorry I haven't finished everything as planned. But at least, I get something working though not ideally good enough.
