# ReLU

ReLU is a common activation function. It stands for Rectified Linear Unit. This activation function helps to introduce 
nonlinearity into our models (commonly seen in neural networks). One big advantage of ReLU is that this function is easy to differentiate, which is helpful for backpropagation. 

## Formula
If the number `x` passed into ReLU is less than 0, return 0. If the number `x` is at least 0, return `x`. 

Note: there are other variants of ReLU such as "Leaky ReLU" that you can play around with as well!

## Documentation
[Check out documentation on ReLU here](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)