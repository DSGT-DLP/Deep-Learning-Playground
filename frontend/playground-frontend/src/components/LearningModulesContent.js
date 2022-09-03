content = {
    modules: [
        {
            title: "What is Machine Learning?",
            sections: [
                {
                    sectionType: "text",
                    content: "Machine Learning is perhaps the most talked-about field in Computer Science right now. It is an incredibly versatile tool that sees applications in a variety of industries and other fields of Computer Science. Yet, it can also be difficult to understand and even intimidating to those who lack experience with it. The heavy amount of math and complicated programming often scares away many of those interested in it. However, it doesn't have to be that way. Through these modules, DLP hopes to provide a thorough introduction to Machine Learning that will give you the ability to use our platform to practice building ML models and learn more about this exciting field."
                },
                {
                    sectionType: "heading1",
                    text: "So what is Machine Learning?"
                },
                {
                    sectionType: "text",
                    content: "Machine Learning refers to a set of models that solve problems by \"learning\" the optimal parameters of the model, rather than those parameters being hardcoded. In other words, instead of every step and parameter of the model being harcoded by a human programmer, the model adjusts its own parameters based on data input in order to maximize the accuracy of the model. There is a large variety of machine learning algorithms out there, ranging from simple Decision Trees to very complex Neural Networks. They can be roughly divided by the types of problem they solve and the way they learn."
                },
                {
                    sectionType: "text",
                    content: "One thing all machine learning models have in common is that in a simple sense, they are all just functions whose exact specifications have been finetuned by a learning process. Since they are all functions, they all take in some input and give some output. In machine learning, this comes in the form of a series of inputs, called \"features\", and one output, \"called a label\"."
                },
                {
                    sectionType: "text",
                    content: "There is a large variety of machine learning algorithms out there, ranging from simple Decision Trees to very complex Neural Networks. They can be roughly divided by the types of problem they solve and the way they learn."
                },
                {
                    sectionType: "heading1",
                    text: "What problems can Machine Learning solve?"
                },
                {
                    sectionType: "text",
                    content: "Problems that machine learning aim to solve are usually categorized as regression problems or classification problems. Regression Problems are those where the model attempts to predict some continous value. An example of a Regression problem would be predicting the price of a house. Since there is technically an infinite amount of values that the price of a house could take, this is a regression problem. On the other hand, a classification problem is one where the model attempts to classify an input as one of a number of a discrete outputs. An example of a classification problem could be a model that takes in an image and attempts to decide if it is a picture of a dog or cat. There are only two possible outputs: dog and cat."
                },
                {
                    sectionType: "mcQuestion",
                    question: "A model that attempted to guess the price of a used car given its make, model, and mileage would be solving what kind of machine learning problem?",
                    correctAnswer: 0,
                    answerChoices: [
                        "Regression",
                        "Classification"
                    ]
                },
                {
                    sectionType: "heading1",
                    text: "Types of Learning"
                },
                {
                    sectionType: "text",
                    content:"All Machine Learning algorithms tweak there own parameters or state in order to optimize their performance. This process is called learning. Machine Learning algorithms can also be classified by the way they learn. The two main categories are Supervised Learning and Unsupervised Learning. Supervised Learning refers to when a model has a separate training phase where the model is given data that has the input and the correct output. The model then compares its own output against the given correct output to measure its accuracy, and adjusts its parameters to increase accuracy. Unsupervised Learning occurs when the algorithm does not recieve any already correct output, but rather trains itself based on input data only."
                },
                {
                    sectionType: "mcQuestion",
                    question: "One machine learning model for classification problems is called k-nearest neighbors. It works by locating k inputs it has previously recieved, determining how it classified the majority of those inputs, and then classifying the new input with the same classification. Is this an example of supervised or unsupervised learning?",
                    correctAnswer: 1,
                    answerChoices: [
                        "Supervised Learning",
                        "Unsupervised Learning"
                    ]
                },
                {
                    sectionType: "text",
                    content: "With these general ideas of machine learning, the next module will begin an exploration of Neural Networks, a type of machine learning model around which deep learning, a subfield of machine learning, is built around."
                }
            ]
        },
        {
            title: "Introduction to the Structure of a Neural Networks",
            sections: [
                {
                    sectionType: "text",
                    content: "A Neural Network is a type of machine learning algorithm that is based on the architecture of the human brain. The study and development of neural networks, sometimes called ANNs(Artifical Neural Networks) is one of the most active fields of machine learning, and is usually referred to as Deep Learning. Just like a human brain, a neural network is comprised of a network of neurons that fire, or activate, based on its inputs. Lets take a look at a neuron in a human brain."
                },
                {
                    sectionType: "image",
                    path: "Diagram of Neuron",
                    caption: ""
                },
                {
                    sectionType: "text",
                    content: "A neuron takes in multiple inputs and produces one output. In a human brain, the outputs and inputs are discrete, a nueron either fires or it doesn't. There is no measure of \"firing intensity\". Herein lies one major difference between human neurons and neural networks: the neurons of neural networks transmit continous values. To see where those values come from, we must examine the structure of a typical ANN."
                },
                {
                    sectionType: "image",
                    path: "Diagram of Neural Network",
                    caption: "An example of the structure of a neural network",
                }
            ]
        }
    ]
}