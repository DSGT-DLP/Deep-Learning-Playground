import neuralnet from '../../images/learn_mod_images/neuralnet.png'
import robot from '../../images/learn_mod_images/robotImage.jpg'
import nueron from '../../images/learn_mod_images/neuron.png'
import nueronWithEquation from '../../images/learn_mod_images/neuronWithEquation.png'
import lossExample from '../../images/learn_mod_images/lossExample.png'
import lossExampleTable from '../../images/learn_mod_images/lossExampleTable.png'
import leakyReLUactivation from '../../images/learn_mod_images/LeakyReLUactivation.png'
import ReLUactivation from '../../images/learn_mod_images/ReLUactivation.png'
import binarystepactivation from '../../images/learn_mod_images/binarystepactivation.png'
import sigmoidactivation from '../../images/learn_mod_images/sigmoidactivation.png'
import tanhactivation from '../../images/learn_mod_images/tanhactivation.png'
import sigmoidfunction from '../../images/learn_mod_images/sigmoidfunction.png'

let content = {
  modules: [
    {
      moduleID: 0,
      title: 'What is Machine Learning?',
      subsections: [
        {
          sectionID: 0,
          title: 'Introduction',
          content: [
            {
              sectionType: 'image',
              path: robot,
              caption:
                'Robots like this one use machine learning to understand how to react to its environment',
              attribution: 'Creative Commons Attribution-Share Alike 4.0 Unported',
              licenseLink: 'https://commons.wikimedia.org/wiki/File:TIAGo%2B%2B_side.jpg',
            },
            {
              sectionType: 'text',
              content:
                "Machine Learning is perhaps the most talked-about field in Computer Science right now. It is an incredibly versatile tool that sees applications in a variety of industries and other fields of Computer Science. Yet, it can also be difficult to understand and even intimidating to those who lack experience with it. The heavy amount of math and complicated programming often scares away many of those interested in it. However, it doesn't have to be that way. Through these modules, DLP hopes to provide a thorough introduction to Machine Learning that will give you the ability to use our platform to practice building ML models and learn more about this exciting field!",
            },
          ],
          points: 10,
        },
        {
          sectionID: 1,
          title: 'So what is Machine Learning?',
          content: [
            {
              sectionType: 'text',
              content:
                'Machine Learning refers to a set of models that solve problems by "learning" the optimal parameters of the model, rather than those parameters being hardcoded. In other words, instead of every step and parameter of the model being harcoded by a human programmer, the model adjusts its own parameters based on data input in order to maximize the accuracy of the model.',
            },
            {
              sectionType: 'text',
              content:
                'One thing all machine learning models have in common is that in a simple sense, they are all just functions whose exact specifications have been finetuned by a learning process. Since they are all functions, they all take in some input and give some output. In machine learning, this comes in the form of a series of inputs, called "features", and one output, called a "label".',
            },
            {
              sectionType: 'text',
              content:
                'There is a large variety of machine learning algorithms out there, ranging from simple Decision Trees to very complex Neural Networks. They can be roughly divided by the types of problem they solve and the way they learn.',
            },
            {
              sectionType: 'mcQuestion',
              question:
                'Say we have a model that predicts the expected traffic on a road based on the time of day, weather, and day of the week. In this model, the weather would be a _______ and the amount of traffic would be the _________.',
              correctAnswer: 1,
              questionID: 0,
              answerChoices: [
                'label, feature',
                'feature, label',
                'label, label',
                'feature, feature',
              ],
            },
          ],
          points: 0,
        },
        {
          sectionID: 2,
          title: 'What problems can Machine Learning solve?',
          content: [
            {
              sectionType: 'text',
              content:
                'Problems that machine learning aim to solve are usually categorized as regression problems or classification problems. Regression Problems are those where the model attempts to predict some continous value. An example of a Regression problem would be predicting the price of a house. Since there is technically an infinite amount of values that the price of a house could take, this is a regression problem. On the other hand, a classification problem is one where the model attempts to classify an input as one of a number of a discrete outputs. An example of a classification problem could be a model that takes in an image and attempts to decide if it is a picture of a dog or cat. There are only two possible outputs: dog and cat.',
            },
            {
              sectionType: 'mcQuestion',
              question:
                'A model that attempted to guess the price of a used car given its make, model, and mileage would be solving what kind of machine learning problem?',
              correctAnswer: 0,
              questionID: 0,
              answerChoices: ['Regression', 'Classification'],
            },
          ],
          points: 10,
        },
        {
          sectionID: 3,
          title: 'Types of Learning',
          content: [
            {
              sectionType: 'text',
              content:
                'All Machine Learning algorithms tweak there own parameters or state in order to optimize their performance. This process is called learning. Machine Learning algorithms can also be classified by the way they learn. The two main categories are Supervised Learning and Unsupervised Learning. Supervised Learning refers to when a model has a separate training phase where the model is given data that has the input and the correct output. The model then compares its own output against the given correct output to measure its accuracy, and adjusts its parameters to increase accuracy. Unsupervised Learning occurs when the algorithm does not recieve any already correct output, but rather trains itself based on input data only.',
            },
            {
              sectionType: 'mcQuestion',
              question:
                'One machine learning model for classification problems is called k-nearest neighbors. It works by locating k inputs it has previously recieved, determining how it classified the majority of those inputs, and then classifying the new input with the same classification. Is this an example of supervised or unsupervised learning?',
              correctAnswer: 1,
              questionID: 0,
              answerChoices: ['Supervised Learning', 'Unsupervised Learning'],
            },
            {
              sectionType: 'text',
              content:
                'With these general ideas of machine learning, the next module will begin an exploration of Neural Networks, a type of machine learning model around which deep learning, a subfield of machine learning, is built around.',
            },
          ],
          points: 10,
        },
      ],
      points: 30,
    },
    {
      moduleID: 1,
      title: 'Introduction to Neural Networks',
      subsections: [
        {
          sectionID: 0,
          title: 'Structure of a Neural Network',
          content: [
            {
              sectionType: 'text',
              content:
                'A Neural Network is a type of machine learning algorithm that is based on the architecture of the human brain. The study and development of neural networks, sometimes called ANNs(Artifical Neural Networks) is one of the most active fields of machine learning, and is usually referred to as Deep Learning. Just like a human brain, a neural network is comprised of a network of neurons that fire, or activate, based on its inputs. Lets take a look at one of these neurons.',
            },
            {
              sectionType: 'image',
              path: nueron,
              caption: 'A typical neuron in a neural network',
              attribution: 'Creative Commons Attribution-Share Alike 3.0 Unported',
              licenseLink:
                'https://commons.wikimedia.org/wiki/File:Multi-Layer_Neural_Network-Vector-Blank.svg',
            },
            {
              sectionType: 'text',
              content:
                'A neuron takes in multiple inputs and produces one output. In a human brain, the outputs and inputs are discrete, a nueron either fires or it doesn\'t. There is no measure of "firing intensity". Herein lies one major difference between human neurons and neural networks: the neurons of neural networks transmit continous values. To see where those values come from, we must examine the structure of a typical ANN.',
            },
            {
              sectionType: 'image',
              path: neuralnet,
              caption: 'An example of the structure of a neural network',
              attribution: 'Creative Commons Attribution-Share Alike 3.0 Unported',
              licenseLink:
                'https://commons.wikimedia.org/wiki/File:Multi-Layer_Neural_Network-Vector-Blank.svg',
            },
            {
              sectionType: 'text',
              content:
                'As shown above, a neural network is built up of layers. There are many different types of layers that manipulate their inputs in different ways. Each layer is made up of a series of nodes, which are like neurons in the brain. Each neuron takes input from every single node in the previous layer, and produces one output value.',
            },
            {
              sectionType: 'text',
              content:
                'The first layer in each neural network is called the input layer, and has an equal number of nodes as the number of features. The last layer is the output layer. For regression problems, the output layer will usually have just one node, and the value that that node outputs is the final output from the model. In classification models, the output layer usually contains a number of nodes corresponding to the number of possible classifications. The classification whose node contains the highest value is then selected as the final output from the model.',
            },
            {
              sectionType: 'frQuestion',
              question:
                'Dave built a neural network that attempts to classify a picture as a dog, cat, or chicken. How many nodes will the last layer of his neural network have?',
              questionID: 0,
              answer: 3,
            },
          ],
          points: 10,
        },
        {
          sectionID: 1,
          title: 'Linear Layers (Perceptrons)',
          content: [
            {
              sectionType: 'text',
              content:
                'Now we know what a neural network looks like generally. But how does it take an input and create an output that makes sense? That magic happens in the layers, of which there are many different types. However, the most common type is a linear layer, also called a perceptron. In fact, a perceptron is sometimes used as a machine learning model on its own, not as part of a neural network.',
            },
            {
              sectionType: 'text',
              content:
                "Let's take a look at a single node. The number of inputs it recieves is equal to the number of nodes in the previous layer, since every node in a layer is mapped to every node in the next layer. That node then produces one output. Let's say we know what we want that output to be. What mathematical operations could we do to produce that value? Well, if we take a hint from the name \"linear layer\", we could multiply each input by a number, called a weight, and then add them together. Let's also add in a constant, called a bias. If you take a look at the equation we created, you can see that it looks a lot like an equation of a line, but with many inputs instead of one.",
            },
            {
              sectionType: 'image',
              path: nueronWithEquation,
              caption: 'The equation for the output Y from one node given inputs X',
              attribution: 'Creative Commons Attribution-Share Alike 3.0 Unported',
              licenseLink:
                'https://commons.wikimedia.org/wiki/File:Multi-Layer_Neural_Network-Vector-Blank.svg',
            },
            {
              sectionType: 'text',
              content:
                "The beauty of neural networks comes in their flexibility. If you imagine just how many ways you can manipulate one node to turn any input into the output you want, and then you remember that it is just one node of many, you can begin to see how powerful neural networks can be. However, the next obvious question that arises is how exactly do we choose the right weights and biases to get the output that we want? That's where the learning part of machine learning comes in. Neural Network learning involves a very mathematical process called gradient descent. This is an advanced topic that will be covered in later modules.",
            },
          ],
          points: 0,
        },
        {
          sectionID: 2,
          title: 'Activation Functions',
          content: [
            {
              sectionType: 'text',
              content:
                "Now that we understand how a linear layer works, we can begin to put together a neural network. As you can imagine, there are many ways that linear layers can be combined with other types of layers to create a neural network. Often, the exact design of a neural network involves a lot of experimentation: seeing what works and what doesn't. In that way, machine learning is actually a highly non-deterministic field of computer science!",
            },
            {
              sectionType: 'text',
              content:
                'However, besides layers, there is another important component of neural networks that we have not yet covered: activation functions. To understand the necessity of an activation function, let us consider the range of a linear layer output. Since it is linear, there is no bounds on what values the layer can output. Sometimes however, we need the values of the output to conform to a certain range. This is often the case with classification models where we want the output to represent probabilities that an input is classified as a certain label correctly. In this case, we would want values between 0 and 1.',
            },
            {
              sectionType: 'text',
              content:
                'This is where activation functions come in. There are many different types, but each type takes in an input and applies a mathematical function to it that constricts its range in some way. Below, we will take a look at some common activation functions.',
            },
            {
              sectionType: 'heading1',
              content: 'Binary Step',
            },
            {
              sectionType: 'image',
              path: binarystepactivation,
              caption: 'The Binary Step Activation Function',
              attribution: 'Creative Commons Attribution-Share Alike 3.0 Unported',
              licenseLink:
                'https://en.wikipedia.org/wiki/Activation_function#/media/File:Activation_binary_step.svg',
            },
            {
              sectionType: 'text',
              content:
                'The Binary Step activation is perhaps the most simple. For any input, if the input is positive, it will output 1, while if the input is negative, it will output 0.',
            },
            {
              sectionType: 'heading1',
              content: 'ReLU',
            },
            {
              sectionType: 'image',
              path: ReLUactivation,
              caption: 'The ReLU Activation Function',
              attribution: 'Creative Commons Attribution-Share Alike 3.0 Unported',
              licenseLink:
                'https://en.wikipedia.org/wiki/Activation_function#/media/File:Activation_rectified_linear.svg',
            },
            {
              sectionType: 'text',
              content:
                'ReLU, or Rectified Linear Unit, is emerging to be one of the most popular activation functions, and it is also fairly simple. If an input is negative, the output will be zero. If input is positive, then the output will just be the input.',
            },
            {
              sectionType: 'heading1',
              content: 'Sigmoid',
            },
            {
              sectionType: 'image',
              path: sigmoidactivation,
              caption: 'The Sigmoid Activation Function',
              attribution: 'Creative Commons Attribution-Share Alike 3.0 Unported',
              licenseLink:
                'https://en.wikipedia.org/wiki/Activation_function#/media/File:Activation_logistic.svg',
            },
            {
              sectionType: 'text',
              content:
                'Sigmoid is a little more complicated than the previous activation functions. It applies a logistic curve to the input, effectively compressing the input to the range between 0 and 1. The equation is pictured below.',
            },
            {
              sectionType: 'image',
              path: sigmoidfunction,
              caption: 'The equation for the sigmoid function',
              attribution: 'Creative Commons Attribution-Share Alike 3.0 Unported',
              licenseLink:
                'https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Activation_tanh.svg/180px-Activation_tanh.svg.png',
            },
            {
              sectionType: 'heading1',
              content: 'Hyperbolic Tangent',
            },
            {
              sectionType: 'image',
              path: tanhactivation,
              caption: 'The Hyperbolic Tangent Activation Function',
              attribution: 'Creative Commons Attribution-Share Alike 3.0 Unported',
              licenseLink:
                'https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Activation_tanh.svg/180px-Activation_tanh.svg.png',
            },
            {
              sectionType: 'text',
              content:
                'Hyperbolic tangent is similar to sigmoid, however it compresses the range of an input to between -1 and 1. As the name suggests, it applies the hyperbolic tangent function in the form f(x) = tanh(x)',
            },
            {
              sectionType: 'heading1',
              content: 'LeakyReLU',
            },
            {
              sectionType: 'image',
              path: leakyReLUactivation,
              caption: 'The Leaky ReLU Activation Function',
              attribution: 'Creative Commons Attribution-Share Alike 3.0 Unported',
              licenseLink:
                'https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Activation_prelu.svg/180px-Activation_prelu.svg.png',
            },
            {
              sectionType: 'text',
              content:
                'Finally, Leaky ReLU is a special variation of ReLU aimed at solving a common problem with ReLU. It is similar to regular ReLU, however, on negative inputs, instead of outputting zero it outputs the input multiplied by a very small number, such as 0.001. This means that the output there is "close enough" to zero, at least relative to the output from positive inputs. If you don\'t know calculus, then the rest of this section might be a little hard to understand, however it is enough to say that using Leaky ReLU often means better model performance than regular ReLU. For those of you who do know calculus, then you might see that the derivative of the negative domain of the function is zero. This is problematic, as part of the process of gradient descent which occurs when training the model involves taking derivatives, and then the weights and biases of the model are adjusted based on that derivative. If the derivative is zero, then the weights and biases won\'t be adjusted. Therefore, instead of 0, the output is made to be a very small number close to 0, which ensures that the model training goes smoothly.',
            },
            {
              sectionType: 'text',
              content:
                'Activation functions play an important role in neural networks. They are combined with layers to create a trainable function that works well for the problem it needs to solve. Oftentimes, they are found just before the output layer so that the output conforms to the expected range, however they can be found anywhere throughout the network.',
            },
            {
              sectionType: 'text',
              content:
                'Now that we have covered most of the structure of the neural network, we need a way to evaluate our neural network! This will be covered in the next module where we talk about loss.',
            },
          ],
          points: 0,
        },
      ],
      points: 10,
    },
    {
      moduleID: 2,
      title: 'Loss',
      subsections: [
        {
          sectionID: 0,
          title: 'What is Loss?',
          content: [
            {
              sectionType: 'text',
              content:
                'Now that we have built a neural network, we need to find a way to measure how well it performs. That is, how often does it make the right prediction and how close are the predictions to the expected value? This measure is called loss, with there existing many different types. Loss is very important, as it provides a measure of how the model performs as it is being trained. It allows to see if the model is actually learning in the direction we want it to. This would be represented by loss decreasing as the model is trained.',
            },
            {
              sectionType: 'text',
              content:
                'Loss is usually calculated based on data that we know the expected output for. This means that we need datapoints that have correct values for both features and labels in order to calculate loss.',
            },
            {
              sectionType: 'text',
              content:
                'The most common type of loss is MSELoss, or Mean Squared Error loss. We will take a look at this in the next section.',
            },
          ],
          points: 0,
        },
        {
          sectionID: 1,
          title: 'Mean Squared Error',
          content: [
            {
              sectionType: 'text',
              content:
                'Mean Squared Error is quite simple. Given a prediction from a model and the expected value from the model at some input, the squared error is simply the squared difference between the predicted value and the expected value. Mean Squared Error is the average squared error of predictions based on multiple different datapoints. Let us take a look at an example.',
            },
            {
              sectionType: 'image',
              path: lossExample,
              caption: 'Predicted and Expected Values',
              attribution: '',
              licenseLink: '',
            },
            {
              sectionType: 'text',
              content:
                "The above chart shows a neural network model given six different inputs. For each input the chart shows the value the model predicted in blue and the value that was expected in orange. For each datapoint, let's calculate the error and the squared error.",
            },
            {
              sectionType: 'image',
              path: lossExampleTable,
              caption: 'Calculated Error and Squared Error',
              attribution: '',
              licenseLink: '',
            },
            {
              sectionType: 'text',
              content:
                'You may wonder why we don\'t just use error, but instead use squared error. This is because we want predictions that are more "incorrect", or further away from their expected value, to influence loss more than predictions that are just slightly off. For example, one datapoint with an error of 6 is more of a problem then six datapoints with an error of 1.',
            },
            {
              sectionType: 'text',
              content:
                'Now that we have calculated all the squared errors, we can just average them to find the MSE of our model.',
            },
            {
              sectionType: 'frQuestion',
              question: 'What is the MSE of the model, rounded to the nearest whole number?',
              questionID: 0,
              answer: 19,
            },
          ],
          points: 10,
        },
        {
          sectionID: 2,
          title: 'Some Last Remarks',
          content: [
            {
              sectionType: 'text',
              content:
                'One last important note to make is that loss is not unique to neural networks. Many other machine learning models use loss as a way to compute their accuracy as well. It is an extremely important part of machine learning, yet luckily fairly easy to understand.',
            },
          ],
          points: 0,
        },
      ],
      points: 10,
    },
  ],
}

export default content
