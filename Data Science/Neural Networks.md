
Neural networks map inputs from one layer to another with the following: $$\mathbf{h}^{(l+1)} = f(W^{(l)} \mathbf{h}^{(l)} + \mathbf{b}^{(l)})$$Suppose we have an input vector $\mathbf{x} = \mathbf{h}^{(0)}$, one actual 'hidden' layer $\mathbf{h}^{(1)}$, and then the output layer $\mathbf{y} = \mathbf{h}^{(2)}$, then:
$$\mathbf{y} = \mathbf{h}^{(2)}$$
$$ \mathbf{y} = f(W^{(1)} \mathbf{h}^{(1)} + \mathbf{b}^{(1)})$$ $$ \mathbf{y} = f(W^{(1)} (f(W^{(0)} \mathbf{h}^{(0)} + \mathbf{b}^{(0)})) + \mathbf{b}^{(1)})$$ $$ \mathbf{y} = f(W^{(1)} (f(W^{(0)} \mathbf{x} + \mathbf{b}^{(0)})) + \mathbf{b}^{(1)})$$ 
### Training: backpropagation




### Connectivity Variations

Convolutional layers are often used.

If the output of a network is fed back into the network as an input, we have a [[Recurrent Neural Networks|recurrent neural network]].

[[Transformer Architecture|Transformers]] are the architecture used for language generation in ChatGPT / GPT-4 etc.