import React from "react";

const Wiki = () => {
  return (
    <>
      <div id="header-section">
        <h1 className="header">Deep Learning Playground Wiki</h1>
      </div>

      <div className="sections" id="basics">
        <h2>Deep Learning Basics</h2>
        <p>
          Deep learning is a field of machine learning that uses a{" "}
          <b>neural network</b> to mimic how human brains (which many consider
          to be a really complicated computer) gain knowledge. This boils down
          to representing the behavior of neurons, which work together in
          response to an input to return an series of electrical signals that
          then becomes an input for another set of neurons.
        </p>
        <p>
          To do this, each neural network in a deep learning model is composed
          of a series of <b>layers</b> that are each composed of a set of
          computer "neurons" that can activate in response to their input.
          Mathematically, the neurons are each assigning a linear <b>weight</b>{" "}
          to all of its inputs that is then fed to all of the neurons of the
          next layer. When all layers are combined, the model takes an input
          layer representing some data and generates an output that is hopefully
          insightful. All layers in between the input layer and the output are
          called "hidden layers."The more hidden layers a model has, the more
          complex it becomes.
        </p>
        <p>
          As the neural network trains, it automatically searches for the best
          weight for each neuron using an <b>optimization algorithm</b> chosen
          when designing the model. In order for the neural network to solve
          more complicated, nonlinear problems, an <b>activation function</b>,
          also picked by the designer, is applied to each layer's output. These
          hyperparameters (which are specified in more detail below) can be
          tweaked to solve problems more effectively.
        </p>
      </div>

      <div className="sections" id="documentation">
        <h2>Deep Learning Playground Documentation</h2>
        <h3>Layers Inventory</h3>
        <h4>
          Choose the activation function of the neural network. Click for more
          information about each function.
        </h4>
        <ul>
          <li>
            <a href="https://pytorch.org/docs/stable/generated/torch.nn.Linear.html">
              Linear
            </a>
          </li>
          <li>
            <a href="https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html">
              ReLU
            </a>
          </li>
          <li>
            <a href="https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html">
              Softmax
            </a>
          </li>
          <li>
            <a href="https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html">
              Sigmoid
            </a>
          </li>
          <li>
            <a href="https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html">
              Tanh
            </a>
          </li>
          <li>
            <a href="https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html">
              LogSoftmax
            </a>
          </li>
        </ul>
        <h3>Deep Learning Parameters</h3>
        <h4>
          Parameters that change behavior about the model itself. Specified
          before the model is trained.
        </h4>
        <ul>
          <li>
            <b>Target Column</b> - The column with the intended outputs of the
            model.
          </li>
          <li>
            <b>Problem Type</b> - Determines whether to predict a discrete set
            of possible outputs <i>(classification)</i> or a continuous set of
            possible outputs <i>(regression)</i>
          </li>
          <li>
            <b>Optimizer Name</b> - The mathematical optimizer used to train the
            neural weights for the model.
          </li>
          <li>
            <b>Epochs</b> - Number of times the model is trained on the data
            set.
          </li>
          <li>
            <b>Shuffle</b> - Randomizes the order of the input data in order to
            reduce bias.
          </li>
          <li>
            <b>Test Size</b> - The proportion of the total data set to be used
            to test the performance of the model.
          </li>
        </ul>
      </div>
    </>
  );
};

export default Wiki;
