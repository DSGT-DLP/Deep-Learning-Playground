# Softmax Layer

The Softmax function is an activation function commonly used. You are taking an array of numbers and converting it into an array of probabilities. This is useful within multiclass classification because it would be nice to figure out what the probability of 
your input being in one of `K` classes is in order to make an "informed judgement/classification".

Since Softmax layer covnerts a list of numbers into a list of probabilities, it follows that the probabilities must add to 1.

## Formula
Suppose you had an array of numbers in the form <!-- $$[z_{1}, z_{2}, z_{3}, z_{4}, ... z_{k}]$$ --> 

<div align="center"><img style="background: white;" src="../../../svg/6a6rfoS7eT.svg"></div>. 

The Softmax formula is:

<!-- $$\frac{e^{z_{i}}}{\sum_{i=1}^{k}(e^{z_{i}})}$$ --> 

<div align="center"><img style="background: white;" src="../../../svg/MNM4eGO01I.svg"></div>

Essentially, you are exponentiating each number (using `e` as the base) and then normalizing by dividing these exponentiated numbers by the total. 

## Documentation
Check out [the documentation on Pytorch's Softmax Layer!](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html)
