
Agents perform actions $a_t$ in an environments, which updates the state $s_{t+1}$ of the environment, and which has a corresponding reward $r_{t+1}$. 

![[rl_image_1.png|500]] 

An agent has a **policy** $\pi(a | s)$ which determines what actions will be performed given the current state of the environment. I.e. it produces a 'value'/probability of each action $a$ given a specified state $s$.

The goal of reinforcement learning is to find the policy $\pi$ which maximizes the 'expected return'.