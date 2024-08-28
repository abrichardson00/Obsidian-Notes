
Humans live in - and evolved through - a world of sensory uncertainty. Our brains are only presented with sensory information, and this is a very incomplete representation of the state of the world around us. We evolved to deal with this uncertainty by generating explanations / causes of what we observe from some internal representation/model of the world. 

We all have subjective experience of this - ...

Brains observe 'evidence' $x$ from the world around them, and want to predict the hypothesis $h$ which explains the observed evidence. It's worth clarifying immediately that terms 'hypothesis', 'causes' and 'explanations' are interchangeable here - $h$ is just some representation of the meaning of a sensory input (a 'hidden layer' of our neural network).  For example, $x = \{\text{raw sound data}\}$, and $h = \{\text{that was a fart}\}$. 

Understanding the causes of our sensory experience ($h$) allows the appropriate guiding of our actions to achieve our goals:

![[learning_causes.svg|700]]

Our ideal model of 'cause likelihood', $p^{*}(h | x)$, is as close to the real $p(h | x)$ as possible - e.g. has minimal prediction error $| h_{(\text{real})} - h_{(\text{predicted})} |$. 

However, minimizing $| h_{(\text{real})} - h_{(\text{predicted})} |$ directly isn't feasible - the real causes exist, but are never observable. Our predicted causes $h_{(predicted)}$ correspond to our mind's internal 'understanding' of what's causing the observed evidence, but we never have the real causes to compare them to.

To deal with this, we could introduce some top-down processing, generating 'expected' evidence $x_{\text{expected}}$, given a predicted cause $h_{\text{predicted}}$. This lets us get our desired learned mapping $x_{\text{real}} \rightarrow h_{\text{predicted}}$ as part of the mapping $x_{\text{real}} \rightarrow h_{\text{predicted}} \rightarrow x_{\text{expected}}$ which can be optimized to minimize $|x_{\text{real}} - x_{\text{expected}} |$. 

![[learning_causes 2.svg|700]]
In other words: we try to minimize surprise, such that no new evidence from the world ($x_{(\text{real})}$) is unexpected.

At this stage, I'm essentially just describing an [[Autoencoders|autoencoders]] - which learn a useful hidden representations $\mathbf{h}$ when trying to minimize the loss from reconstructing $\mathbf{x}$. Another small detail: it's safe to assume we have constraints on $\mathbf{h}$ such that an autoencoding network doesn't just learn identity mappings. 


### Recurrent Sensory Processing

Our brains don't compute $h_{(\text{predicted})}$ from an autoencoder however - we can't really assume brains hold duplicate representations of both $x_{\text{(real)}}$ and $x_{(\text{expected})}$ which it then compares to learn $h_{\text{(predicted)}}$. 

Instead, it is likely computed through a [[Recurrent Neural Networks|recurrent neural network]] with bottom-up connectivity ($x_{\text{real}} \rightarrow h_{\text{predicted}}$), and top-down connectivity ($x_{\text{expected}} \leftarrow h_{\text{predicted}}$) - however, both $x_{\text{(real)}}$ and $x_{(\text{expected})}$ are represented in the one state: $x$. 

A better way to think of this shared $x$ state, is by calling it the perceived input $x_{(\text{perceived})}$. Our perceived input $x_{\text{(perceived)}}$ is still affected by the real input $x_{\text{(real)}}$, but it is also affected by $h_{(\text{predicted})}$. I think it's now more appropriate to rename $x_{\text{(real)}}$ as $u$ - since we should think of it as an input (or bias) to the network rather than a state stored by the neurons. So we have $u_{(\text{real})} \rightarrow x_{(\text{perceived})} \leftrightarrow h_{\text{(predicted)}}$. More explicitly, our RNN updates the state of our neurons $\mathbf{r}$ with:
$$\mathbf{r}^{(t+1)} = f(W\mathbf{r}^{(t)} + \mathbf{u}), \text{where:}$$
$$\mathbf{r}^{(t+1)} = \begin{bmatrix} \mathbf{x} \\ \mathbf{h} \end{bmatrix}^{(t+1)} = f(\begin{bmatrix} \mathbf{0} & W_{\mathbf{h}\rightarrow \mathbf{x}} \\ W_{\mathbf{x}\rightarrow \mathbf{h}}   & \mathbf{0} \end{bmatrix} \begin{bmatrix} \mathbf{x} \\ \mathbf{h} \end{bmatrix}^{(t)} + \begin{bmatrix} \mathbf{u} \\ \mathbf{0} \end{bmatrix}) $$ 
![[perception_rnn_1.svg|450]]

There's an interesting insight gained from thinking about the brain's computation like this. The sensory evidence perceived by the brain's RNN, $x_{\text{(perceived)}}$, depends on both the 'ground truth' sensory input $u$, and the higher order 'understanding of the perceived world', $h_{(\text{predicted})}$. This lower-level input representation $x_{(\text{perceived})}$ which we genuinely see / feel, depends on more than just the real sensory input. In other words, perception is a controlled hallucination.  

##### Big picture:

In deep-learning, feed-forward networks can be used to learn functional mappings from input to output. Like the brain, these networks use hierarchical processing to handle new / incomplete 'sensory' data. However, the learning processes are quite different.

Our brains obviously learn differently to supervised learning methods - there's no 'output layer' in our brain where we can compare produced output with training labels. They're more similar to unsupervised deep-learning methods such as autoencoders - however, instead of having separate sets of neurons for the encoder and decoder, there is a shared set of neurons with bottom-up and top-down connectivity doing the 'encoding' and 'decoding'. Learning then involves ensuring that expectations fed down from the higher-level representations align with the sensory data being received through lower-level layers upwards. 

Our real brains are RNNs, because we have these feedback 'top-down' connections which allow our high-level extracted features to influence our lower-level perception.

### Bayesian Brain and Schizophrenia

The interpretations of lower level input ($x$) are constrained by higher level representations ($h$).

![[Pasted image 20231122143212.png|500]]



![[active_inference.svg]]


 $p(h | x) = \frac{p(x | h) p(h)}{p(x)}$
$p(h_{3} | x) = p(h_{3} | h_{2}) p(h_{2}| x)$.  



$$p(c_1 | e) = \frac{p(e | c_1) p(c_1)}{p(e)}$$

