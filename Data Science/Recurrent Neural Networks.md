
Note: RNNs are [[Dynamical Systems]] - see [[Master's Thesis]] and [[Computational Neuroscience]]



### Feed-forward Networks as a dynamical system

[[Neural Networks]] map inputs from one layer to another with the following: $$\mathbf{h}^{(l+1)} = f(W^{(l)} \mathbf{h}^{(l)} + \mathbf{b}^{(l)})$$Suppose we have an input vector $\mathbf{x} = \mathbf{h}^{(0)}$, one actual 'hidden' layer $\mathbf{h}^{(1)}$, and then the output layer $\mathbf{y} = \mathbf{h}^{(2)}$, then:
$$\mathbf{y} = \mathbf{h}^{(2)}$$
$$ \mathbf{y} = f(W^{(1)} \mathbf{h}^{(1)} + \mathbf{b}^{(1)})$$ $$ \mathbf{y} = f(W^{(1)} (f(W^{(0)} \mathbf{h}^{(0)} + \mathbf{b}^{(0)})) + \mathbf{b}^{(1)})$$ $$ \mathbf{y} = f(W^{(1)} (f(W^{(0)} \mathbf{x} + \mathbf{b}^{(0)})) + \mathbf{b}^{(1)})$$Also, for simplicity lets drop the bias terms $\mathbf{b}$, leaving us with the following mapping from input $\mathbf{x}$ to output $\mathbf{y}$:  $$ \mathbf{y} = f(W^{(1)} (f(W^{(0)} \mathbf{x}))$$
Now, suppose we concatenate the state of all 'neurons' - input, hidden and output - into one vector: 
$$\mathbf{r} = \begin{bmatrix} \mathbf{x} \\ \mathbf{h} \\ \mathbf{y} \end{bmatrix}$$
We could achieve the same computation with one 'global' system by transforming an input state with only $\mathbf{x}$ ($\mathbf{r}^{(0)} = \begin{bmatrix} \mathbf{x} & \mathbf{0} & \mathbf{0} \end{bmatrix}^{\top}$) to an output state only including $\mathbf{y}$, ($\mathbf{r}^{(2)} = \begin{bmatrix} \mathbf{0} & \mathbf{0} & \mathbf{y} \end{bmatrix}^{\top}$):
$$ \mathbf{r}^{(l)} = f(W \mathbf{r}^{(l-1)}), \text{where} \ \   W = \begin{bmatrix} \mathbf{0} & \mathbf{0} & \mathbf{0} \\ W^{\mathbf{x} \rightarrow \mathbf{h}} & \mathbf{0} & \mathbf{0} \\ \mathbf{0} & W^{\mathbf{h} \rightarrow \mathbf{y}} & \mathbf{0} \end{bmatrix}$$  $$\mathbf{r}^{(0)} = \begin{bmatrix} \mathbf{x} \\ \mathbf{0} \\ \mathbf{0} \end{bmatrix}$$$$ \rightarrow 
\mathbf{r}^{(1)} =  f(W \mathbf{r}^{(0)}) = \begin{bmatrix} \mathbf{0} \\ f(W^{\mathbf{x} \rightarrow \mathbf{h}}\mathbf{x}) \\ \mathbf{0} \end{bmatrix} = \begin{bmatrix} \mathbf{0} \\ \mathbf{h} \\ \mathbf{0} \end{bmatrix}$$
$$\rightarrow 
\mathbf{r}^{(2)} =  f(W \mathbf{r}^{(1)}) = \begin{bmatrix} \mathbf{0} \\ \mathbf{0} \\ f(W^{\mathbf{h} \rightarrow \mathbf{y}} \mathbf{h}) \end{bmatrix} = \begin{bmatrix} \mathbf{0} \\ \mathbf{0} \\ \mathbf{y} \end{bmatrix}$$ 
When implementing this method, one wouldn't store multiple $\mathbf{r}^{(l)}$ as 'layers' of the computation. Instead the layers $\mathbf{h}^{(l)}$ are stored in one vector $\mathbf{r}$, and the update $f(W \mathbf{r})$ has to be applied multiple times to calculate $\mathbf{x} \rightarrow \mathbf{h}^{(1)} \rightarrow ... \rightarrow \mathbf{h}^{(N_{h})} \rightarrow \mathbf{y}$.

So neural networks, even feed-forward neural nets, are dynamical systems which update the set of values $\mathbf{r}$ at each timestep with: $\mathbf{r}^{(t+1)} = f(W \mathbf{r}^{(t)})$ 

Or one could also say the system is updated with $\mathbf{r} \rightarrow \mathbf{r} + \frac{d\mathbf{r}}{dt}$ with the change $\frac{d\mathbf{r}}{dt} = f(W \mathbf{r}) - \mathbf{r}$.

### Different architectures are variations on the global $W$

The matrix $W$ is the connectivity / adjacency matrix that 'connects' any 2 neurons. So in theory, all possible networks are represented by different constraints on $W$.

##### Feed-forward Networks

In the case of feed-forward networks, this corresponds to a lower triangular form W (without the diagonal identity elements too). The graph specified by adjacency matrix W is a directed acyclic graph - there are no loops, since this would define a recurrent neural network instead.


```
                            (from)
  1 (in)        | - - - - - - - - - - - - - - -
  2 (in)        | - - - - - - - - - - - - - - -
  3 (in)        | - - - - - - - - - - - - - - -
  4 (in)        | - - - - - - - - - - - - - - -
  5 (in)        | - - - - - - - - - - - - - - -
  6             | x x x x x - - - - - - - - - -
  7             | x x x x x - - - - - - - - - -
  8             | x x x x x - - - - - - - - - -  (to)
  9             | x x x x x - - - - - - - - - -
  10            | x x x x x - - - - - - - - - -
  11 (out)      | - - - - - x x x x x - - - - -
  12 (out)      | - - - - - x x x x x - - - - -
  13 (out)      | - - - - - x x x x x - - - - -
  14 (out)      | - - - - - x x x x x - - - - -
  15 (out)      | - - - - - x x x x x - - - - -
```



An autoencoder could have the following adjacency matrix:
```
                                  (from)
  1 (in)        | - - - - - - - - - - - - - - - - - - - - - -
  2 (in)        | - - - - - - - - - - - - - - - - - - - - - -
  3 (in)        | - - - - - - - - - - - - - - - - - - - - - -
  4 (in)        | - - - - - - - - - - - - - - - - - - - - - -
  5 (in)        | - - - - - - - - - - - - - - - - - - - - - -
  6 (in)        | - - - - - - - - - - - - - - - - - - - - - -
  7             | x x x x x x - - - - - - - - - - - - - - - -
  8             | x x x x x x - - - - - - - - - - - - - - - -
  9             | x x x x x x - - - - - - - - - - - - - - - -
  10            | x x x x x x - - - - - - - - - - - - - - - -
  11            | - - - - - - x x x x - - - - - - - - - - - -    (to)
  12            | - - - - - - x x x x - - - - - - - - - - - -
  13            | - - - - - - - - - - x x - - - - - - - - - -
  14            | - - - - - - - - - - x x - - - - - - - - - -
  15            | - - - - - - - - - - x x - - - - - - - - - -
  16            | - - - - - - - - - - x x - - - - - - - - - -
  17 (out)      | - - - - - - - - - - - - x x x x - - - - - -
  18 (out)      | - - - - - - - - - - - - x x x x - - - - - -
  19 (out)      | - - - - - - - - - - - - x x x x - - - - - -
  20 (out)      | - - - - - - - - - - - - x x x x - - - - - -
  21 (out)      | - - - - - - - - - - - - x x x x - - - - - -
  22 (out)      | - - - - - - - - - - - - x x x x - - - - - -
```

![[autoencoder2.png|300]]


## Recurrent Neural Nets

Suppose we have a simple feedforward net with 2 input, output, and hidden neurons, and then we added recurrent connections from individual hidden $\mathbf{h}$ neurons to themselves (i.e. 3rd and 4th diagonal elements of $W$):

```
  1 (in)        | - - - - - -                   | - - - - - -
  2 (in)        | - - - - - -                   | - - - - - -
  3             | x x - - - -        ->         | x x x - - -    
  4             | x x - - - -                   | x x - x - -
  5 (out)       | - - x x - -                   | - - x x - -
  6 (out)       | - - x x - -                   | - - x x - -
```

Assuming the recurrent weights are 1, we could have $W = \begin{bmatrix} 0 & 0 & 0 \\ W^{\mathbf{x}\rightarrow \mathbf{h}} & I & 0 \\ 0 & W^{\mathbf{h} \rightarrow \mathbf{y}} & 0 \end{bmatrix}$. 

We can still compute $\mathbf{y}$ from $\mathbf{x}$, but the evolution of state $\mathbf{r}$ would be: 
$$\mathbf{r} = \begin{bmatrix} \mathbf{x} \\ \mathbf{0} \\ \mathbf{0} \end{bmatrix} \rightarrow \begin{bmatrix} \mathbf{0} \\ \mathbf{h} \\ \mathbf{0} \end{bmatrix} \rightarrow \begin{bmatrix} \mathbf{0} \\ \mathbf{h} \\ \mathbf{y} \end{bmatrix} 
\rightarrow \begin{bmatrix} \mathbf{0} \\ \mathbf{h} \\ \mathbf{y} \end{bmatrix}
\rightarrow \begin{bmatrix} \mathbf{0} \\ \mathbf{h} \\ \mathbf{y} \end{bmatrix}
\rightarrow \ ... $$
Recall our dynamical system is updating with $\mathbf{r} \rightarrow \mathbf{r} + \frac{d\mathbf{r}}{dt}$, with the change $\frac{d\mathbf{r}}{dt} = f(W \mathbf{r}) - \mathbf{r}$. The state with $\mathbf{y}$ is a fixed point - i.e. we have $\frac{d\mathbf{r}}{dt} = \mathbf{0}$. 

On the other hand, consider the evolution of state $\mathbf{r}$ for the feed-forward network discussed earlier:
$$\mathbf{r} = \begin{bmatrix} \mathbf{x} \\ \mathbf{0} \\ \mathbf{0} \end{bmatrix} \rightarrow \begin{bmatrix} \mathbf{0} \\ \mathbf{h} \\ \mathbf{0} \end{bmatrix} \rightarrow \begin{bmatrix} \mathbf{0} \\ \mathbf{0} \\ \mathbf{y} \end{bmatrix} 
\rightarrow \begin{bmatrix} \mathbf{0} \\ \mathbf{0} \\ \mathbf{0} \end{bmatrix}
\rightarrow \begin{bmatrix} \mathbf{0} \\ \mathbf{0} \\ \mathbf{0} \end{bmatrix}
\rightarrow \ ... $$

When you consider neural networks as dynamical system updating the state $\mathbf{r}$ over time, feed-forward networks are a subset of all the possible networks, and they always have their fixed point $\mathbf{r} = \mathbf{0}$ (when including biases, this isn't actually true - but it still, for any input $\mathbf{x}$, tends to one fixed point which is only a function of the biases).

## Training - Backpropagation through time

Consider the following recurrent network: 

$W = \begin{bmatrix} 0 & 0 & 0 \\ W^{\mathbf{x} \rightarrow \mathbf{h}} & W^{\mathbf{h} \rightarrow \mathbf{h}} & 0 \\ 0 & W^{\mathbf{h} \rightarrow \mathbf{y}} & 0\end{bmatrix}$, i.e. $\mathbf{h}^{(t)} = f(W^{\mathbf{x} \rightarrow \mathbf{h}}\mathbf{x}^{(t)} + W^{\mathbf{h}\rightarrow \mathbf{h}} \mathbf{h}^{(t-1)})$ 

![[rnn1.png|300]]
Given a sequence of inputs $\{ \mathbf{x}^{(1)} \ ... \ \mathbf{x}^{(t)} \}$, the global state of the system $\mathbf{r}$ is updated over time to produce the output sequence $\{ \mathbf{y}^{(1)} \ ... \ \mathbf{y}^{(t)} \}$, or additionally, some longer output sequence $\{ \mathbf{y}^{(1)} \ ... \ \mathbf{y}^{(t)} \textcolor{red}{\ ... \ \mathbf{y}^{(\text{future})}} \}$:

$$\mathbf{r} 
= \begin{bmatrix} \mathbf{x}^{(1)} \\ \mathbf{0} \\ \mathbf{0} \end{bmatrix}
\rightarrow \begin{bmatrix} \mathbf{x}^{(2)} \\ \mathbf{h}^{(1)} \\ \mathbf{0} \end{bmatrix}
\rightarrow \begin{bmatrix} \mathbf{x}^{(3)} \\ \mathbf{h}^{(2)} \\ \mathbf{y}^{(1)} \end{bmatrix}
\rightarrow \ ... \ 
\rightarrow \begin{bmatrix} \mathbf{x}^{(t)} \\ \mathbf{h}^{(t-1)} \\ \mathbf{y}^{(t-2)} \end{bmatrix}
\rightarrow \begin{bmatrix} \mathbf{0} \\ \mathbf{h}^{(t)} \\ \mathbf{y}^{(t-1)} \end{bmatrix}
\rightarrow \begin{bmatrix} \mathbf{0} \\ \textcolor{red}{\mathbf{h}^{(t+1)}} \\ \mathbf{y}^{(t)} \end{bmatrix}
\rightarrow \begin{bmatrix} \mathbf{0} \\ \textcolor{red}{\mathbf{h}^{(t+2)}} \\ \textcolor{red}{\mathbf{y}^{(t+1)}} \end{bmatrix}
\rightarrow \ ...
$$
To backpropagate errors to update $W$, one actually converts a $W$ with recurrent weights to a feed-forward connectivity matrix $W^{FF}$. Then regular backpropagation / automatic reverse-mode differentiation is applied to this feed-forward network. 

As an example consider the case where we want to map $\{ \mathbf{x}^{(1)}, \ \mathbf{x}^{(2)}, \ \mathbf{x}^{(3)} \}$ to $\{ \mathbf{y}^{(1)}, \ \mathbf{y}^{(2)}, \ \mathbf{y}^{(3)} \}$. This is achieved with:
$$\mathbf{r} 
= \begin{bmatrix} \mathbf{x}^{(1)} \\ \mathbf{0} \\ \mathbf{0} \end{bmatrix}
\rightarrow \begin{bmatrix} \mathbf{x}^{(2)} \\ \mathbf{h}^{(1)} \\ \mathbf{0} \end{bmatrix}
\rightarrow \begin{bmatrix} \mathbf{x}^{(3)} \\ \mathbf{h}^{(2)} \\ \mathbf{y}^{(1)} \end{bmatrix}
\rightarrow \begin{bmatrix} \mathbf{0} \\ \mathbf{h}^{(3)} \\ \mathbf{y}^{(2)} \end{bmatrix}
\rightarrow \begin{bmatrix} \mathbf{0} \\ \textcolor{red}{\mathbf{h}^{(4)}} \\ \mathbf{y}^{(3)} 
\end{bmatrix}
$$
If we change the arrows to represent which values are computed from what values in each timestep, we have:
$$\mathbf{r} 
= \begin{bmatrix} \mathbf{x}^{(1)} \\ \mathbf{0} \\ \mathbf{0} \end{bmatrix}
\begin{matrix} \searrow \\ \rightarrow \\ \  \end{matrix} 
 \begin{bmatrix} \mathbf{x}^{(2)} \\ \mathbf{h}^{(1)} \\ \mathbf{0} \end{bmatrix}
\begin{matrix} \searrow \\ \rightarrow \\ \searrow \end{matrix}  
 \begin{bmatrix} \mathbf{x}^{(3)} \\ \mathbf{h}^{(2)} \\ \mathbf{y}^{(1)} \end{bmatrix}
\begin{matrix} \searrow \\ \rightarrow \\ \searrow \end{matrix} 
 \begin{bmatrix} \mathbf{0} \\ \mathbf{h}^{(3)} \\ \mathbf{y}^{(2)} \end{bmatrix}
\begin{matrix} \ \\ \rightarrow \\ \searrow \end{matrix}  
 \begin{bmatrix} \mathbf{0} \\ \textcolor{red}{\mathbf{h}^{(4)}} \\ \mathbf{y}^{(3)} 
\end{bmatrix}
$$
Imagine we considered the different values of $\mathbf{h}$ as completely different states entirely - with their own connectivity matrices $W^{\mathbf{x}^{(t)} \rightarrow \mathbf{h^{(t)}}}$ and $W^{\mathbf{h}^{(t-1)} \rightarrow \mathbf{h}^{(t)}}$, and we might have $W^{\mathbf{h}^{(1)} \rightarrow \mathbf{h}^{(2)}} \neq W^{\mathbf{h}^{(2)} \rightarrow \mathbf{h}^{(3)}}$. Then the computation graph above is indeed a feed-forward network, mapping one concatenated input vector $\mathbf{x} = \begin{bmatrix} \mathbf{x}^{(1)} & \mathbf{x}^{(2)} & \mathbf{x}^{(3)} \end{bmatrix}^{\top}$ to the output $\mathbf{y} = \begin{bmatrix} \mathbf{y}^{(1)} & \mathbf{y}^{(2)} & \mathbf{y}^{(3)} \end{bmatrix}^{\top}$:

$$\begin{matrix} \mathbf{x}^{(1)} & \mathbf{x}^{(2)} & \mathbf{x}^{(3)}\\ \\
\textcolor{green}{\downarrow} \\ \\
\mathbf{h}^{(1)} &  \textcolor{green}{\downarrow} \\
\ \textcolor{lightblue}{\searrow} &  \\ 
& \mathbf{h}^{(2)} & \textcolor{green}{\downarrow} \\
\textcolor{red}{\downarrow} & \ \textcolor{lightblue}{\searrow} \\
& & \mathbf{h}^{(3)} \\
& \textcolor{red}{\downarrow} \\ \\
& & \textcolor{red}{\downarrow} \\ \\
\mathbf{y}^{(1)} & \mathbf{y}^{(2)} & \mathbf{y}^{(3)}



\end{matrix}$$
When drawing out this graph of the computations, each arrow corresponds to applying some $[\text{next}] = f(W^{\text{prev} \rightarrow \text{next}} [\text{prev}])$, e.g. first green arrow represents computation of $\mathbf{h}^{(1)} = f(W^{\mathbf{x}^{(1)} \rightarrow \mathbf{h}^{(1)}} \mathbf{x}^{(1)})$.

Given a training pair of input $\mathbf{x}$ and output $\mathbf{y}$, backpropagation of the error of the network's output (i.e. error $E$ is some magnitude of $\mathbf{y}_{\text{training}} - \mathbf{y}_{\text{output}}$) is used to update the weights and make the network perform slightly better for such examples in the future. This reverse-mode automatic differentiation will give changes $\frac{\partial W}{\partial E}$ for every matrix (every arrow above) - but this is for the constructed feed-forward version of the recurrent networks computation. To get desired changes for the RNN, the changes for each type of matrix (indicated by colour in arrows above and text below) are just added together. Explicitly, we have:
$$\frac{\partial W^{\mathbf{x}^{(t)} \rightarrow \mathbf{h}^{(t)}}}{\partial E} = \textcolor{green}{\frac{\partial W^{\mathbf{x}^{(1)} \rightarrow \mathbf{h}^{(1)}}}{\partial E} + \frac{\partial W^{\mathbf{x}^{(2)} \rightarrow \mathbf{h}^{(2)}}}{\partial E} + \frac{\partial W^{\mathbf{x}^{(3)} \rightarrow \mathbf{h}^{(3)}}}{\partial E}}$$
$$\frac{\partial W^{\mathbf{h}^{(t-1)} \rightarrow \mathbf{h}^{(t)}}}{\partial E} = \textcolor{lightblue}{\frac{\partial W^{\mathbf{h}^{(1)} \rightarrow \mathbf{h}^{(2)}}}{\partial E} + \frac{\partial W^{\mathbf{h}^{(2)} \rightarrow \mathbf{h}^{(3)}}}{\partial E}}$$
$$\frac{\partial W^{\mathbf{h}^{(t)} \rightarrow \mathbf{y}^{(t)}}}{\partial E} = \textcolor{red}{\frac{\partial W^{\mathbf{h}^{(1)} \rightarrow \mathbf{y}^{(1)}}}{\partial E} + \frac{\partial W^{\mathbf{h}^{(2)} \rightarrow \mathbf{y}^{(2)}}}{\partial E} + \frac{\partial W^{\mathbf{h}^{(3)} \rightarrow \mathbf{y}^{(3)}}}{\partial E}}$$
### Why bother with RNNs if we'll unroll it anyway?

We've shown that the computation performed by an RNN can be performed by an 'unrolled' feed-forward network on a set of larger concatenated $\mathbf{x}$, $\mathbf{h}$ and $\mathbf{y}$ neurons. So what's the point of using RNNs if a feed-forward network could just perform the same thing. In fact, surely the feed-forward network we constructed is better because it has more matrices which could handle values differently due to their position in the sequence?

Ultimately, the answer to this is "it depends" - one can use feed-forward networks for handling sequences, however they can only process and produce a fixed length of sequence.

Of course, one can just produce one output token of a sequence $\mathbf{y}^{(i)}$ given some length of input sequence, and then re-apply the feed-forward model on different offsets of the input sequence to produce different terms of the output sequence. **In fact**, this is how [[Transformer Architecture|transformer models]] like ChatGPT work - they are not RNNs. Also, these feed-forward text generation models can generate arbitrarily long sequences because the single token they generate, $\mathbf{y}^{(t+1)}$, is the predicted next word after the input sequence of words $\{ \mathbf{x}^{(1)} \ ... \ \mathbf{x}^{(t)} \}$ - hence one can always just append $\mathbf{y}^{(t+1)}$ to the input sequence and then generate the next token, $\mathbf{y}^{(t+2)}$ etc. 

So while feed-forward models can generate arbitrarily long sequences, this is restricted to problems where you want to predict the next token of the input sequence (so that the predicted token can actually be used as the next input token). RNNs on the other hand, don't need this elongation of input sequence - instead, after the inputs stop, the hidden state is still updated and outputs $\mathbf{y}^{t}$ onwards are produced.

#### RNN advantages

 - can process arbitrarily long input ***sequence histories with a smaller number of re-used parameters*** (weights) compared to a feed-forward network which requires more weights to process a larger history of tokens. Inference with RNNs can certainly be cheaper with reduced parameter set, but I don't know if training is necessarily cheaper since we unroll RNNs.
 - can ***generate arbitrarily long output sequences*** after an input sequence terminates, even if output sequence is of different format to input sequence. Arbitrarily long feed-forward output sequence generation would require same input / output format since they always require the inputs. 
#### RNN disadvantages

 - in practise, RNNs often struggle processing long histories through the recurrent weights. I.e. the hidden state struggles to include any information from previous inputs. Hence recent large language models are not RNNs. 

### Hopfield Networks


