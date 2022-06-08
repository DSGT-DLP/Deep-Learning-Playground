# Deep Learning Basics
Deep learning is a field of machine learning that uses a **neural network** to mimic how human brains (which many consider to be a really complicated computer) gain knowledge. This boils down to representing the behavior of neurons, which work together in response to an input to return an series of electrical signals that then becomes an input for another set of neurons.
To do this, each neural network in a deep learning model is composed of a series of **layers** that are each composed of a set of computer "neurons"that can activate in response to their input. Mathematically, the neurons are each assigning a linear **weight** to all of its inputs that is then fed to all of the neurons of the next layer. When all layers are combined, the model takes an input layer representing some data and generates an output that is hopefully insightful. All layers in between the input layer and the output are called "hidden layers."The more hidden layers a model has, the more complex it becomes.
As the neural network trains, it automatically searches for the best weight for each neuron using an **optimization algorithm** specified when designing the model. In order for the neural network to solve more complicated, nonlinear problems, an **activation function** also specified by the designer is applied to each layer's output. These hyperparameters (which are specified in more detail below) can be tweaked to solve problems more effectively.
# Deep Learning Playground Documentation
## Layers Inventory
### Choose the activation function of the neural network.
https://medium.com/fintechexplained/neural-network-layers-75e48d71f392
- 	Linear - https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
- 	ReLU - https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
- 	Softmax - https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
- 	Sigmoid - https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html
- 	Tanh - https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html
- 	LogSoftmax - https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html

## Deep Learning Parameters
### Parameters that change behavior about the model itself. Specified before the model is trained.
- 	Target Column - The column with the intended outputs of the model.
- 	Problem Type
- 		Classification - Predicts an output over a discrete set.
- 		Regression - Predicts an output over a continuous set.
- 	Optimizer Name - 
- 	Epochs - Number of times the model is trained on the data set.
- 	Shuffle - Randomizes the order of the input data in order to reduce bias.
- 	Test Size - The proportion of the total data set to be used to test the performance of the model.