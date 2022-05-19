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
