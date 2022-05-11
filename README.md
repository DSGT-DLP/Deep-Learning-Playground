# Deep-Learning-Playground
Web Application where people new to Deep Learning can input a dataset and toy around with basic Pytorch modules

## Conda Env Setup
`conda env create -f environment.yml` in the `/conda` directory

## Backend Infra
`python driver.py` in the `/backend` directory

## Backend Architecture
```
📦backend
 ┣ 📜constants.py - list of helpful constants
 ┣ 📜data.csv - data csv file for use in the playground
 ┣ 📜dataset.py - read in the dataset through URL or file upload
 ┣ 📜driver.py - run the backend (entrypoint script)
 ┣ 📜input_parser.py - parse the user specified pytorch model
 ┣ 📜loss.py - loss function enum
 ┣ 📜model.py - torch model based on user specifications from drag and drop
 ┣ 📜optimizer.py - what optimizer to use (ie: SGD or Adam for now)
 ┣ 📜trainer.py - train the model on the dataset
 ┣ 📜utils.py - utility functions that could be helpful
 ┗ 📜__init__.py
```
