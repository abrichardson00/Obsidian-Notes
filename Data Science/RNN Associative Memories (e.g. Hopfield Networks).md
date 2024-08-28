

Recurrent neural networks (RNNs) are [[Dynamical Systems]], so while changing inputs to the network will directly change the output, the output values will also change over time (even with no / constant inputs). The state of dynamical systems over time can exhibit chaotic dynamics, have values tend to infinity, or reach attractor states. 

Recall our [[Recurrent Neural Networks|RNN]] updates it's state with $\mathbf{r} \rightarrow \mathbf{r} + \frac{d\mathbf{r}}{dt}$, with the change $\frac{d\mathbf{r}}{dt} = f(W \mathbf{r}) - \mathbf{r}$.

Attractor states are particularly interesting since they can be used as associative memories. One can configure RNNs such that certain 'memories' are attractor states of the network. Then when appropriate input sets the state of the RNN close enough to this memory state, $\mathbf{r}$ then changes over time towards the attractor - recalling this 'memory'. Such attractor dynamics certainly play a role in how the brain works - associative memory is well studied in psychology.

Attractor states have $\frac{d\mathbf{r}}{d t} = 0$, and are stable - i.e. have a concave $\frac{d\mathbf{r}}{d t}$ landscape (a small temporary nudge away from the fixed point will be re-attracted back to the fixed point).

The question becomes: how can we configure RNNs for desired memory recall? We know our attractor states need to have $\frac{d\mathbf{r}}{d t} = 0$, but how do we set $W$ such that these fixed points are for values of $\mathbf{r}$ that correspond to our desired 'memories'.

Backpropagation through time is typically used with RNNs, which will produce an RNN with dynamics attempting to satisfy an input to output mapping. This is a powerful technique that will produce arbitrary dynamics - not just fixed points, though these may indeed occur. 

However, for pure memory-recall other methods are available. 


## Hopfield Networks



![[Pasted image 20231115105246.png]]