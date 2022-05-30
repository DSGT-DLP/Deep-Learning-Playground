# Linear Layer

This layer in Pytorch is how you add a "hidden layer" in a neural network. When you say for example `nn.Linear(10, 4)`, what you are doing is multiplying a `1x10` matrix by a `10x4` matrix to get a `1x4` matrix. This `10x4` matrix contains weights that
need to be learned along with a "bias" term (additive bias). In linear algebra terms, the Linear layer is doing `xW^{T} + b` where `W` is the weight matrix (ie: the `10x4` in our example).

## Example Usage in Pytorch
```
x = torch.randn(10, 20) #10x20 matrix
lin_layer = nn.Linear(20, 5) #create linear layer that's 20x5 matrix
lin_layer(x) #run/apply linear layer on input x
```

## More Information
Check out [the documentation on Pytorch's Linear Layer!](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)