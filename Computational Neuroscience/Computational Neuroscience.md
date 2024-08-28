---
tags:
  - neuroscience
  - machine-learning
---

[[Neural Networks]]
[[Dynamical Systems]]
[[Master's Thesis]]


## Dynamical System Models of Neural Networks

The brain is incredibly complicated, so when modelling it we make lots of approximations. 
#### Biology recap: 

Neurons receive input electrical signals from other neurons through their dendrites, and once the input crosses some threshold an 'action-potential' (aka a spike, or the neuron 'fires') is generated from the neuron's soma (central part). This spike - it's output electrical signal - is propagated along the neuron's axon and becomes input to other neurons. 

![[Pasted image 20240113192323.png]]

Even this description is a huge approximation - the branching dendrites likely perform complex logical computation on the inputs. Also the criteria for making a neuron fire is more complicated than a simple input threshold.

Spicy many-to-many connectivity between our billions of neurons gives rise to our human experience: perceiving things, remembering things, feeling emotions and so on.

#### Approximations:

The most accurate neural network model would model the voltages of individual neurons, modelling the dynamics of each neuron's action potentials.

Spiking network models approximate the firing as a boolean on/off - i.e. 1 for the moment when a neuron fires, and 0 otherwise. 

![[Pasted image 20240113194104.png]]

Firing rate models approximate things one step further, modelling the changing approximate frequency of which these spikes occur. We are now modelling continuous values, which makes things much nicer. At this point our 'neuron' interpretation is essentially the same as in deep-learning models. 



#### Firing Rate Models:

We define the way in which our firing rates $\mathbf{r}$ will change over time ($\frac{d\mathbf{r}}{dt}$) according to $W$ - the way in which the neurons are all connected:

$\frac{d\mathbf{r}}{dt} = -\mathbf{r} + \boldsymbol{\phi}(\mathbf{u} + W \mathbf{r})$

Our 'simulation of the brain' is just updating our values of $\mathbf{r}$ over time with $\mathbf{r} \rightarrow \mathbf{r} + \frac{d\mathbf{r}}{dt}$ according to the equation above.

## Mechanisms of Associative Memory 

## Bayesian Brain

