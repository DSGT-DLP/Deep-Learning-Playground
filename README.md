# Deep-Learning-Playground
Web Application where people new to Deep Learning can input a dataset and toy around with basic Pytorch modules through a drag and drop interface.

## Conda Env Setup
* `conda env create -f environment.yml` in the `/conda` directory

* Updating an environment: `conda env update -f environment.yml` in the `/conda` directory
## Backend Infra
`python driver.py` in the `/backend` directory

The backend supports training of a deep learning model and/or a classical ML model!
## Backend Architecture
```
📦backend
 ┣ 📜constants.py - list of helpful constants
 ┣ 📜data.csv - data csv file for use in the playground
 ┣ 📜dl_eval.py - Evaluation functions for deep learning models in Pytorch (eg: accuracy, loss, etc)
 ┣ 📜dataset.py - read in the dataset through URL or file upload
 ┣ 📜driver.py - run the backend (entrypoint script)
 ┣ 📜input_parser.py - parse the user specified pytorch model
 ┣ 📜loss.py - loss function enum
 ┣ 📜model.py - torch model based on user specifications from drag and drop
 ┣ 📜optimizer.py - what optimizer to use (ie: SGD or Adam for now)
 ┣ 📜dl_trainer.py - train a deep learning model on the dataset
 ┣ 📜ml_trainer.py - train a classical machine learning learning model on the dataset
 ┣ 📜utils.py - utility functions that could be helpful
 ┣ 📜webdriver.py - Selenium Webdriver script to take a deep learning model in the form of an onnx file and upload it to netron.app to visualize (Work in Progress)
 ┗ 📜__init__.py
```

## Frontend 

### Startup Instructions

1. For complete functionality with the backend, first, start the backend using the instructions above. The backend will live in http://localhost:5000/run

2. Then in a separate terminal, start the frontend development server. After installing [nodeJS v16](https://nodejs.org/en/download/), run the following commands:
```
cd frontend\playground-frontend
npm install
npm start
```
3. Then, go to http://localhost:3000/

### How to Add New Layers
Currently, there are three layers implemented in this playgroud—Linear, ReLU, and Softmax. A developer can easily add in a new layer to be used by the user through:
1. Go to [settings.js](./frontend/playground-frontend/src/settings.js)
2. Put in (* = required):
    - `display_name`*: Name of layer to be displayed to user
    - `object_name`*: Layer object to be passed into the backend, e.g., `nn.linear(...)`
    - `parameters`: An array of JS objects with at least the display name of the parameters for the layer object
