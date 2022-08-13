import React, { useState } from "react";
import DEMO_VIDEO from "../../images/demo_video.gif";
import softmax_eq from "./softmax_equation.png";

const displayChange = (display, setdisplay) => {
  if (display === "block") {
    setdisplay("none");
  } else {
    setdisplay("block");
  }
};

const render_layer_info = (layer_info) => {
  return (
    <li style={{ marginBottom: "10px" }}>
      <button
        onClick={() =>
          displayChange(layer_info.displayState, layer_info.setDisplayState)
        }
        className="layer-info-button"
      >
        {layer_info.title}
      </button>

      <div
        style={{ display: layer_info.displayState }}
        className="wiki-content"
      >
        {layer_info.body}
      </div>
    </li>
  );
};

const render_all_layers = (layer_wiki) => {
  layer_wiki = layer_wiki.sort((a, b) => (a.title > b.title ? 1 : -1));
  const body = [];

  for (let i = 0; i < layer_wiki.length; i++) {
    console.log(layer_wiki[i]);
    body.push(render_layer_info(layer_wiki[i]));
  }
  console.log(body);
  return <ul>{body}</ul>;
};

const Wiki = () => {
  const [linearDisp, setLinearDisp] = useState("none");
  const [softmax, setSoftmax] = useState("none");
  const [relu, setRelu] = useState("none");

  const layer_wiki = [
    {
      title: "Linear Layer",
      body: (
        <>
          <p>
            This layer in Pytorch is how you add a "hidden layer" in a neural
            network. When you say for example `nn.Linear(10, 4)`, what you are
            doing is multiplying a `1x10` matrix by a `10x4` matrix to get a
            `1x4` matrix. This `10x4` matrix contains weights that need to be
            learned along with a "bias" term (additive bias). In linear algebra
            terms, the Linear layer is doing `xW^(T) + b` where `W` is the
            weight matrix (ie: the `10x4` in our example).
          </p>

          <h5>Example Usage in Pytorch</h5>

          <pre>
            <code>
              {`
          x = torch.randn(10, 20)  # 10x20 matrix 
          lin_layer = nn.Linear(20, 5)  # create linear layer that's 20x5 matrix
          lin_layer(x) # run/apply linear layer on input x
          `}
            </code>
          </pre>

          <h5>Documentation</h5>

          <p>
            Check out{" "}
            <a href="https://pytorch.org/docs/stable/generated/torch.nn.Linear.html">
              the documentation on Pytorch's Linear Layer!
            </a>
          </p>
        </>
      ),
      displayState: linearDisp,
      setDisplayState: setLinearDisp,
    },
    {
      title: "ReLU",
      body: (
        <>
          <p>
            ReLU is a common activation function. It stands for Rectified Linear
            Unit. This activation function helps to introduce nonlinearity into
            our models (commonly seen in neural networks). One big advantage of
            ReLU is that this function is easy to differentiate, which is
            helpful for backpropagation.{" "}
          </p>

          <h5>Formula</h5>

          <p>
            If the number `x` passed into ReLU is less than 0, return 0. If the
            number `x` is at least 0, return `x`.
          </p>
          <p>
            Note: there are other variants of ReLU such as "Leaky ReLU" that you
            can play around with as well!
          </p>

          <h5>Documentation</h5>

          <p>
            Check out{" "}
            <a href="https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html">
              the documentation on ReLU Layer here!
            </a>
          </p>
        </>
      ),
      displayState: relu,
      setDisplayState: setRelu,
    },
    {
      title: "Softmax",
      body: (
        <>
          <p>
            The Softmax function is an activation function commonly used. You
            are taking an array of numbers and converting it into an array of
            probabilities. This is useful within multiclass classification
            because it would be nice to figure out what the probability of your
            input being in one of `K` classes is in order to make an "informed
            judgement/classification". Since Softmax layer covnerts a list of
            numbers into a list of probabilities, it follows that the
            probabilities must add to 1.
          </p>

          <p>
            Suppose you had an array of numbers in the form `[z_1, z_2, ... z_
            k]`
          </p>

          <p>The Softmax formula is:</p>

          <img
            src={softmax_eq}
            alt="Softmax equation"
            style={{ maxHeight: 200, marginInline: "auto" }}
          />

          <p>
            Essentially, you are exponentiating each number (using `e` as the
            base) and then normalizing by dividing these exponentiated numbers
            by the total.
          </p>

          <h5>Documentation</h5>

          <p>
            Check out{" "}
            <a href="https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html">
              the documentation on Pytorch's Softmax Layer!
            </a>
          </p>
        </>
      ),
      displayState: softmax,
      setDisplayState: setSoftmax,
    },
  ];

  return (
    <>
      <div id="header-section">
        <h1 className="header">Deep Learning Playground Wiki</h1>
      </div>

      <div className="sections" id="basics">
        <h2>Live Demo</h2>
        <img
          src={DEMO_VIDEO}
          alt="GIF showing a demo of the playground"
          loading="lazy"
          style={{ width: "100%" }}
        />
      </div>

      <div className="sections" id="basics">
        <h2>What is Deep Learning?</h2>
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
        <p>
          Choose the activation function of the neural network. Click for more
          information about each function.
        </p>

        {render_all_layers(layer_wiki)}

        <h3>Deep Learning Parameters</h3>
        <p>
          Parameters that change behavior about the model itself. Specified
          before the model is trained.
        </p>
        <ul style={{ gap: "10px", display: "grid" }}>
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
          <li>
            <b>Batch Size</b> - Number of datapoints processed each time before
            the model is updated.
          </li>
        </ul>
      </div>
    </>
  );
};

export default Wiki;
