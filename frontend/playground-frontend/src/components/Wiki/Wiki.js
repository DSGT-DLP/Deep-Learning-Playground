import React, { useState } from 'react'
import DEMO_VIDEO from '../../images/demo_video.gif'
import softmax_eq from './softmax_equation.png'
import tanh_eq from './tanh_equation.png'
import tanh_plot from './tanh_plot.png'
import sigmoid_eq from './sigmoid_equation.png'
import dropout_dg from './dropout_diagram.png'
import conv2dgif from './conv2d.gif'
import conv2dgif2 from './conv2d2.gif'
import maxpool2dgif from './maxpool2d.gif'
import avgmaxpoolgif from './avgpool_maxpool.gif'
import batchnorm from './batchnorm_diagram.png'

const displayChange = (display, setdisplay) => {
  if (display === 'block') {
    setdisplay('none')
  } else {
    setdisplay('block')
  }
}

const render_layer_info = (layer_info) => {
  const changeClassName = () => {
    const layer_info_button = document.getElementById(layer_info.id)
    if (layer_info_button.classList.contains('collapsed-layer')) {
      layer_info_button.classList.remove('collapsed-layer')
      layer_info_button.classList.add('expanded-layer')
    } else {
      layer_info_button.classList.remove('expanded-layer')
      layer_info_button.classList.add('collapsed-layer')
    }
  }
  let body = []
  for (let i = 0; i < layer_info.docs.length; i++) {
    body.push(
      <li
        key={layer_info.docs[i].layer_name}
        style={{ marginBottom: '10px', display: layer_info.displayState }}
      >
        <button
          onClick={() =>
            displayChange(layer_info.docs[i].displayState, layer_info.docs[i].setDisplayState)
          }
          className='layer-info-button'
        >
          {layer_info.docs[i].layer_name}
        </button>

        <div style={{ display: layer_info.docs[i].displayState }} className='wiki-content'>
          {layer_info.docs[i].body}
        </div>
      </li>,
    )
  }
  return (
    <>
      <button
        onClick={() => {
          changeClassName()
          displayChange(layer_info.displayState, layer_info.setDisplayState)
        }}
        className='layer-outer-button'
        style={{
          marginBottom: layer_info.displayState === 'none' ? '10px' : 0,
        }}
      >
        {layer_info.title}
      </button>
      <ul>{body}</ul>
    </>
  )
}

const render_all_layer_info = (layer_wiki) => {
  const body = []
  for (const layer_element of layer_wiki) {
    body.push(
      <ul className='collapsed-layer' id={layer_element.id} key={layer_element.id}>
        <li key={layer_element.title}>{render_layer_info(layer_element)}</li>
      </ul>,
    )
  }
  return body
}

const Wiki = () => {
  const [common, setCommon] = useState('none')
  const [linearDisp, setLinearDisp] = useState('none')
  const [conv, setConv] = useState('none')
  const [softmax, setSoftmax] = useState('none')
  const [relu, setRelu] = useState('none')
  const [nla, setnla] = useState('none')
  const [outerrelu, setouterrelu] = useState('none')
  const [tanh, settanh] = useState('none')
  const [sigmoid, setSigmoid] = useState('none')
  const [conv2d, setconv2d] = useState('none')
  const [maxpool2d, setMaxpool2d] = useState('none')
  const [adaptiveAvgPool2d, setAdaptiveAvgPool2d] = useState('none')
  const [dropout, setDropout] = useState('none')
  const [batchNorm2d, setBatchNorm2d] = useState('none')

  const layer_wiki = [
    {
      title: 'Common Layers',
      id: 'common-layers',
      docs: [
        {
          layer_name: 'Linear Layer',
          body: (
            <>
              <p>
                This layer in Pytorch is how you add a "hidden layer" in a neural network. When you
                say for example `nn.Linear(10, 4)`, what you are doing is multiplying a `1x10`
                matrix by a `10x4` matrix to get a `1x4` matrix. This `10x4` matrix contains weights
                that need to be learned along with a "bias" term (additive bias). In linear algebra
                terms, the Linear layer is doing `xW^(T) + b` where `W` is the weight matrix (ie:
                the `10x4` in our example).
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
                Check out{' '}
                <a href='https://pytorch.org/docs/stable/generated/torch.nn.Linear.html'>
                  the documentation on Pytorch's Linear Layer!
                </a>
              </p>
            </>
          ),
          displayState: linearDisp,
          setDisplayState: setLinearDisp,
        },
        {
          layer_name: 'Dropout',
          body: (
            <>
              <p>
                When we are training deep learning models, one common problem that can come up is
                that the model overfits (ie: the model learns from the data it's fed in well, but
                cannot generalize well to unseen data). We need to find a way to mitigate this
                problem. One common pattern used is to implement a Dropout layer. This layer will
                randomly zero out the weight of some of the neurons that feed into the Dropout
                layer. In other words, we are "crippling" our neural network in a probabilistic
                fashion
              </p>

              <p> Below is the dropout layer in action (notice the X marks) </p>
              <img
                src={dropout_dg}
                alt='Dropout diagram'
                style={{ maxHeight: 200, marginInline: 'auto' }}
              />

              <h5>Documentation</h5>
              <p>
                Check out{' '}
                <a href='https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html'>
                  the documentation on Pytorch's Dropout Layer!
                </a>
              </p>
            </>
          ),
          displayState: dropout,
          setDisplayState: setDropout,
        },
      ],
      displayState: common,
      setDisplayState: setCommon,
    },
    {
      title: 'Non-linear Activations (weighted sum, nonlinearity)',
      id: 'non-linear-activations-weighted',
      docs: [
        {
          layer_name: 'ReLU',
          body: (
            <>
              <p>
                ReLU is a common activation function. It stands for Rectified Linear Unit. This
                activation function helps to introduce nonlinearity into our models (commonly seen
                in neural networks). One big advantage of ReLU is that this function is easy to
                differentiate, which is helpful for backpropagation.{' '}
              </p>

              <h5>Formula</h5>

              <p>
                If the number `x` passed into ReLU is less than 0, return 0. If the number `x` is at
                least 0, return `x`.
              </p>
              <p>
                Note: there are other variants of ReLU such as "Leaky ReLU" that you can play around
                with as well!
              </p>

              <h5>Documentation</h5>

              <p>
                Check out{' '}
                <a href='https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html'>
                  the documentation on ReLU Layer here!
                </a>
              </p>
            </>
          ),
          displayState: relu,
          setDisplayState: setRelu,
        },
        {
          layer_name: 'Tanh',
          body: (
            <>
              <p>
                Tanh (hyperbolic tangent) is a common activation function. This function helps to
                introduce nonlinearity into out models (which helps make our neural networks better
                able to learn more complex patterns in our data). The advantage is that the negative
                inputs will be mapped strongly negative and the zero inputs will be mapped near zero
                in the tanh graph.
              </p>
              <p>The Tanh formula is:</p>

              <img
                src={tanh_eq}
                alt='Tanh equation'
                style={{ maxHeight: 200, marginInline: 'auto' }}
              />

              <p> Below is a plot for tanh to get an intuition for what this function is doing: </p>

              <img
                src={tanh_plot}
                alt='Tanh plot'
                style={{ maxHeight: 300, marginInline: 'auto' }}
              />

              <h5>Documentation</h5>

              <p>
                Check out{' '}
                <a href='https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html'>
                  the documentation on Tanh Activation function here!
                </a>
              </p>
            </>
          ),
          displayState: tanh,
          setDisplayState: settanh,
        },
        {
          layer_name: 'Sigmoid',
          body: (
            <>
              <p>
                Sigmoid is a common activation function. This function is commonly used within
                neural networks. Contrary to the Tanh activation function, sigmoid will squish very
                negative value near 0 and squish very positive values near 1. While sigmoid is an
                easy to differentiate function, it suffers from what's called the "vanishing
                gradient problem". Essentially, when your neural network is trying to learn from
                your data, it is performing optimizations to update the weights for each
                "hyperparameter" (think about manipulating the dials and knobs to minimize your
                loss). The neural network's optimization involves calculating gradients (AKA "rates
                of change") and if the rates of change are too small, then the neural network ends
                up updating its weights by a very tiny amount. As shown in the below plot, the
                vanishing gradient problem tends to occur when applying the sigmoid on very negative
                or very positive values.
              </p>

              <p> Plot and Equation of the sigmoid function is shown below</p>
              <img
                src={sigmoid_eq}
                alt='Sigmoid plot'
                style={{ maxHeight: 300, marginInline: 'auto' }}
              />

              <h5>Documentation</h5>

              <p>
                Check out{' '}
                <a href='https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html'>
                  the documentation on Sigmoid Activation function here!
                </a>
              </p>
            </>
          ),
          displayState: sigmoid,
          setDisplayState: setSigmoid,
        },
      ],
      displayState: outerrelu,
      setDisplayState: setouterrelu,
    },
    {
      title: 'Non-Linear Activations',
      id: 'non-linear-activations',
      docs: [
        {
          layer_name: 'Softmax',
          body: (
            <>
              <p>
                The Softmax function is an activation function commonly used. You are taking an
                array of numbers and converting it into an array of probabilities. This is useful
                within multiclass classification because it would be nice to figure out what the
                probability of your input being in one of `K` classes is in order to make an
                "informed judgement/classification". Since Softmax layer covnerts a list of numbers
                into a list of probabilities, it follows that the probabilities must add to 1.
              </p>

              <p>Suppose you had an array of numbers in the form `[z_1, z_2, ... z_ k]`</p>

              <p>The Softmax formula is:</p>

              <img
                src={softmax_eq}
                alt='Softmax equation'
                style={{ maxHeight: 200, marginInline: 'auto' }}
              />

              <p>
                Essentially, you are exponentiating each number (using `e` as the base) and then
                normalizing by dividing these exponentiated numbers by the total.
              </p>

              <h5>Documentation</h5>

              <p>
                Check out{' '}
                <a href='https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html'>
                  the documentation on Pytorch's Softmax Layer!
                </a>
              </p>
            </>
          ),
          displayState: softmax,
          setDisplayState: setSoftmax,
        },
      ],
      displayState: nla,
      setDisplayState: setnla,
    },
    {
      title: 'Convolution Layers',
      id: 'convolution-layers',
      docs: [
        {
          layer_name: 'Conv2d',
          body: (
            <>
              <p>
                Convolution layers are commonly used within image datasets. Convolution is a fancy
                word for "sliding window" (you specify the dimensions of the windows you want to
                look at). Images can be thought of as a rectangular matrix of pixels where each
                pixel communicates the RGB value to describe the particular color for a particular
                location in the image. Conv2d is a layer that takes in a matrix of weights and goes
                through all possible "windows" in a sliding fashion. In each window, we take the
                pixel values, multiply them by the corresponding weights in the matrix (elementwise)
                and then sum up those products to populate in our result matrix (AKA weighted sum of
                pixels). This layer is best understood through visual demonstrations (as shown
                below)
              </p>

              <p>Helpful gifs to understand conv2d</p>

              <img
                src={conv2dgif}
                alt='Conv 2d gif #1'
                style={{ maxHeight: 400, marginInline: 'auto' }}
              />

              <img
                src={conv2dgif2}
                alt='Conv 2d gif #2'
                style={{ maxHeight: 500, marginInline: 'auto' }}
              />

              <h5>Documentation</h5>

              <p>
                Check out{' '}
                <a href='https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html'>
                  the documentation on Conv2d layer here!
                </a>
              </p>
            </>
          ),
          displayState: conv2d,
          setDisplayState: setconv2d,
        },
        {
          layer_name: 'Max Pool 2d',
          body: (
            <>
              <p>
                This layer traverses all possible MxN windows in the image (you specify the
                dimension of the windows to look at). Instead of doing a weighted sum of pixels,
                each window you see, you will simply take the maximum pixel value and populate the
                resulting matrix.
              </p>

              <p>Max pool in action</p>

              <img
                src={maxpool2dgif}
                alt='Max pool 2d gif'
                style={{ maxHeight: 400, marginInline: 'auto' }}
              />

              <h5>Documentation</h5>

              <p>
                Check out{' '}
                <a href='https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html'>
                  the documentation on MaxPool2d layer here!
                </a>
              </p>
            </>
          ),
          displayState: maxpool2d,
          setDisplayState: setMaxpool2d,
        },
        {
          layer_name: 'Adaptive Average Pool 2d',
          body: (
            <>
              <p>
                Adaptive Average Pool 2d is a layer that allows you to specify the dimensions of the
                output. This layer will infer what window size should be used to get your output
                dimension. The auto-inference of the window size is the "Adaptive" aspect, but the
                "Average Pool" aspect will take an average (arithmetic mean) of the pixel values for
                each "window".
              </p>

              <p>Adaptive Avg Pool and Max Pool 2d in action</p>

              <img
                src={avgmaxpoolgif}
                alt='Max pool and Avg pool side by side'
                style={{ maxHeight: 400, marginInline: 'auto' }}
              />

              <h5>Documentation</h5>

              <p>
                Check out{' '}
                <a href='https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html'>
                  the documentation on Adaptive Avg Pool 2d layer here!
                </a>
              </p>
            </>
          ),
          displayState: adaptiveAvgPool2d,
          setDisplayState: setAdaptiveAvgPool2d,
        },
        {
          layer_name: 'Batch Norm 2d',
          body: (
            <>
              <p>
                In data science, you generally tend to work with datasets that have "multiple
                features" (ie: multiple attributes that can influence the target you want to predict
                or classify). It would be very simplistic to assume that all features (ie: inputs)
                are always on the same scale. Before feeding data into a model, it is considered
                good practice to normalize your features such that all inputs are on the same scale
                and you can do an "apples to apples" comparison. In deep learning, it would be nice
                to normalize the input coming from previous layers before feeding into the next
                layer such that the model can reach optimal weights in less iterations. Batch norm
                handles this for us!
              </p>

              <p>Below is a diagram showing what happens under the hood in Batch norm</p>

              <img
                src={batchnorm}
                alt='Batch norm 2d'
                style={{ maxHeight: 400, marginInline: 'auto' }}
              />

              <p>
                What you really need to take away from this diagram is the following: So, you have
                input data coming in from the previous layer in your model (since deep learning
                involves layers of computations). Batch norm is doing the following:
              </p>
              <ol>
                <br />
                <li>Calculate the mean and variance for each feature/column. </li>
                <br />
                <li>
                  Knowing the mean and variance of each feature allows you to standardize your
                  features (in the same way you calculate what's called a "z-score"). Now, your
                  normalized data will have mean 0 and variance 1
                </li>
                <br />
                <li>
                  Scale and Shift is performed. Note that you don't necessarily specify the scale
                  and shift parameters. Batch Norm actually can learn the optimal scale/shift
                  parameters just like other weights in your deep learning model. This is part of
                  the magic of Batch Norm!
                </li>
                <br />
                <li>
                  Instead of storing the mean and variance for each feature/column, Batch Norm
                  actually stores the moving average for mean and variance for efficiency. These
                  values are generally good approximations for the real mean/variance of your
                  features. The moving average for mean and variance allows for Batch Norm to learn
                  the optimal scale/shift parameters during backpropagation in deep learning
                </li>
              </ol>
              <br />

              <h5>Documentation</h5>
              <a href='https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739'>
                Check out this helpful article for intuitively understanding Batch Norm
              </a>
              <p>
                Check out{' '}
                <a href='https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html'>
                  the documentation on Adaptive Avg Pool 2d layer here!
                </a>
              </p>
            </>
          ),
          displayState: batchNorm2d,
          setDisplayState: setBatchNorm2d,
        },
      ],
      displayState: conv,
      setDisplayState: setConv,
    },
  ]

  return (
    <div id='wiki'>
      <div id='header-section'>
        <h1 className='header'>Deep Learning Playground Wiki</h1>
      </div>

      <div className='sections'>
        <h2>Live Demo</h2>
        <img
          src={DEMO_VIDEO}
          alt='GIF showing a demo of the playground'
          loading='lazy'
          style={{ width: '100%' }}
        />
      </div>

      <div className='sections' id='basics'>
        <h2>What is Deep Learning?</h2>
        <p>
          Deep learning is a field of machine learning that uses a <b>neural network</b> to mimic
          how human brains (which many consider to be a really complicated computer) gain knowledge.
          This boils down to representing the behavior of neurons, which work together in response
          to an input to return an series of electrical signals that then becomes an input for
          another set of neurons.
        </p>
        <p>
          To do this, each neural network in a deep learning model is composed of a series of{' '}
          <b>layers</b> that are each composed of a set of computer "neurons" that can activate in
          response to their input. Mathematically, the neurons are each assigning a linear{' '}
          <b>weight</b> to all of its inputs that is then fed to all of the neurons of the next
          layer. When all layers are combined, the model takes an input layer representing some data
          and generates an output that is hopefully insightful. All layers in between the input
          layer and the output are called "hidden layers."The more hidden layers a model has, the
          more complex it becomes.
        </p>
        <p>
          As the neural network trains, it automatically searches for the best weight for each
          neuron using an <b>optimization algorithm</b> chosen when designing the model. In order
          for the neural network to solve more complicated, nonlinear problems, an{' '}
          <b>activation function</b>, also picked by the designer, is applied to each layer's
          output. These hyperparameters (which are specified in more detail below) can be tweaked to
          solve problems more effectively.
        </p>
      </div>

      <div className='sections' id='documentation'>
        <h2>Deep Learning Playground Documentation</h2>
        <h3>Layers Inventory</h3>
        <p>Click for more information about each layer.</p>

        {/* {render_all_layers(layer_wiki)} */}
        {render_all_layer_info(layer_wiki)}

        <h3>Deep Learning Parameters</h3>
        <p>
          Parameters that change behavior about the model itself. Specified before the model is
          trained.
        </p>
        <ul style={{ gap: '10px', display: 'grid' }}>
          <li>
            <b>Target Column</b> - The column with the intended outputs of the model.
          </li>
          <li>
            <b>Problem Type</b> - Determines whether to predict a discrete set of possible outputs{' '}
            <i>(classification)</i> or a continuous set of possible outputs <i>(regression)</i>
          </li>
          <li>
            <b>Optimizer Name</b> - The mathematical optimizer used to train the neural weights for
            the model.
          </li>
          <li>
            <b>Epochs</b> - Number of times the model is trained on the data set.
          </li>
          <li>
            <b>Shuffle</b> - Randomizes the order of the input data in order to reduce bias.
          </li>
          <li>
            <b>Test Size</b> - The proportion of the total data set to be used to test the
            performance of the model.
          </li>
          <li>
            <b>Batch Size</b> - Number of datapoints processed each time before the model is
            updated.
          </li>
        </ul>
      </div>
    </div>
  )
}

export default Wiki
