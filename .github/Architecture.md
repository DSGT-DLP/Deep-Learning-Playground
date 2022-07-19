# Architecture

## Backend Architecture

```
📦backend
 ┣ 📂aws_helpers
 ┃ ┗ 📂dynamo_db_utils
 ┃ ┃ ┣ 📜base_db.py - General Dynamo DB Utility class that other Dynamo DB can inherit
 ┃ ┃ ┗ 📜status_db.py - General Dynamo DB table for status
 ┣ 📂common
 ┃ ┣ 📜constants.py - list of helpful constants
 ┃ ┣ 📜dataset.py - read in the dataset through URL or file upload
 ┃ ┣ 📜default_datasets.py - store logic to load in default datasets from scikit-learn
 ┃ ┣ 📜email_notifier.py - ENdpoint to send email notification of training results via API Gateway + AWS SES
 ┃ ┣ 📜loss_functions.py - loss function enum
 ┃ ┣ 📜optimizer.py - what optimizer to use (ie: SGD or Adam for now)
 ┃ ┗ 📜utils.py - utility functions that could be helpful
 ┣ 📂dl
 ┃ ┣ 📜dl_eval.py - Evaluation functions for deep learning models in Pytorch (eg: accuracy, loss, etc)
 ┃ ┣ 📜dl_model.py - torch model based on user specifications from drag and drop
 ┃ ┣ 📜dl_model_parser.py - parse the user specified pytorch model
 ┃ ┣ 📜dl_trainer.py - train a deep learning model on the dataset
 ┃ ┗ 📜pretrained.py - Functionality to support user training pretrained models (eg: alexnet, resnet, vgg16, etc) via timmodels + fast AI 
 ┣ 📂ml
 ┃ ┗ 📜ml_trainer.py - train a classical machine learning learning model on the dataset
 ┣ 📜data.csv - data csv file for use in the playground
 ┣ 📜driver.py - run the backend (entrypoint script)
 ┣ 📜epoch_times.csv
 ┣ 📜test.py
 ┣ 📜webdriver.py
 ┗ 📜__init__.py
```

## Frontend Architecture

```
📦playground-frontend
 ┣ 📂layer_docs
 ┃ ┣ 📜Linear.md - Doc for Linear layer
 ┃ ┣ 📜ReLU.md - Doc for ReLU later
 ┃ ┣ 📜Softmax.md - Doc for Softmax layer
 ┃ ┗ 📜softmax_equation.png - PNG file of Softmax equation
 ┣ 📂public
 ┃ ┣ 📜dlp-logo.ico - DLP Logo
 ┃ ┣ 📜dlp-logo.png - DLP Logo
 ┃ ┣ 📜dlp-logo.svg - DLP Logo
 ┃ ┣ 📜index.html - Base HTML file that will be initially rendered
 ┃ ┣ 📜manifest.json - Default React file for choosing icon based on
 ┃ ┗ 📜robots.txt
 ┣ 📂src
 ┃ ┣ 📂backend_outputs
 ┃ ┃ ┣ 📜model.pt - Last model.pt output
 ┃ ┃ ┗ 📜my_deep_learning_model.onnx - Last ONNX file output
 ┃ ┣ 📂components
 ┃ ┃ ┣ 📂About
 ┃ ┃ ┃ ┗ 📜About.js - Primary About page component
 ┃ ┃ ┣ 📂Feedback
 ┃ ┃ ┃ ┗ 📜Feedback.js - Primary Feedback page component
 ┃ ┃ ┣ 📂Footer
 ┃ ┃ ┃ ┣ 📜Footer.css - CSS file for Footer
 ┃ ┃ ┃ ┗ 📜Footer.js - Primary Footer page component
 ┃ ┃ ┣ 📂general
 ┃ ┃ ┃ ┗ 📜TitleText.js - Renders a simple header for a small subsection title
 ┃ ┃ ┣ 📂helper_functions
 ┃ ┃ ┃ ┗ 📜TalkWithBackend.js - Sends ML/DL parameters to the backend and receives the backend
 ┃ ┃ ┣ 📂Home
 ┃ ┃ ┃ ┣ 📜AddedLayer.js - Renders an added layer container in the topmost row
 ┃ ┃ ┃ ┣ 📜AddNewLayer.js - Renders a fillable layer container for the user to drag LayerChoice components into, where the new layer input will be filled by an AddedLayer component
 ┃ ┃ ┃ ┣ 📜BackgroundLayout.js - Renders a light blue horizontally-stretched row to serve as a background to contain other components in one subsection in the Homepage
 ┃ ┃ ┃ ┣ 📜CodeSnippet.js - Renders the code snippet container
 ┃ ┃ ┃ ┣ 📜CSVInputFile.js - Renders the CSV file input contents (if any)
 ┃ ┃ ┃ ┣ 📜CSVInputURL.js - Renders the CSV URL input contents (if any)
 ┃ ┃ ┃ ┣ 📜DropDown.js - Renders the drop down components using react-select package
 ┃ ┃ ┃ ┣ 📜EmailInput.js - Renders the email input form
 ┃ ┃ ┃ ┣ 📜Input.js - Renders the Parameters for the machine/deep learning tools (often DropDown input components)
 ┃ ┃ ┃ ┣ 📜LayerChoice.js - Renders a layer container for all the possible layers, for the user to drag from into the AddNewLayer component
 ┃ ┃ ┃ ┣ 📜RectContainer.js - Renders a stylizable fixed-sized rectangle for the layers
 ┃ ┃ ┃ ┣ 📜Results.js - Renders the results after a train session, or a simple text if no train sessions have been done or a session has failed with the backend's message
 ┃ ┃ ┃ ┗ 📜TrainButton.js - Renders the Train button, clicking which will call the backend with the ML/DL parameters
 ┃ ┃ ┣ 📂Navbar
 ┃ ┃ ┃ ┗ 📜Navbar.js - Primary Navbar page component
 ┃ ┃ ┣ 📂Wiki
 ┃ ┃ ┃ ┣ 📜softmax_equation.png - Softmax equation screenshot for reference in Wiki
 ┃ ┃ ┃ ┗ 📜Wiki.js - Primary Wiki page component
 ┃ ┃ ┗ 📜index.js - Centralized location to import any components from outside of this ./components directory
 ┃ ┣ 📂images
 ┃ ┃ ┣ 📂logos
 ┃ ┃ ┃ ┣ 📂dlp_branding
 ┃ ┃ ┃ ┃ ┣ 📜dlp-logo.png - DLP Logo, duplicate of files in public, but essential as the frontend can't read public
 ┃ ┃ ┃ ┃ ┗ 📜dlp-logo.svg - DLP Logo, duplicate of files in public, but essential as the frontend can't read public
 ┃ ┃ ┃ ┣ 📜aws-logo.png
 ┃ ┃ ┃ ┣ 📜dsgt-logo-dark.png
 ┃ ┃ ┃ ┣ 📜dsgt-logo-light.png
 ┃ ┃ ┃ ┣ 📜dsgt-logo-white-back.png
 ┃ ┃ ┃ ┣ 📜flask-logo.png
 ┃ ┃ ┃ ┣ 📜pandas-logo.svg
 ┃ ┃ ┃ ┣ 📜python-logo.png
 ┃ ┃ ┃ ┣ 📜pytorch-logo.png
 ┃ ┃ ┃ ┗ 📜react-logo.png
 ┃ ┃ ┗ 📜demo_video.gif - GIF tutorial of a simple classification training session
 ┃ ┣ 📜App.css - General CSS file
 ┃ ┣ 📜App.js - Base React file
 ┃ ┣ 📜constants.js - Constants for the frontend
 ┃ ┣ 📜Home.js - Main project file that renders the Home frontpage
 ┃ ┣ 📜index.js - Calls the App.js file to render to the .root DOM element
 ┃ ┣ 📜iris.csv - Sample CSV data
 ┃ ┗ 📜settings.js - List of layers and parameters supported in this playground
 ┣ 📜.gitignore
 ┣ 📜package-lock.json
 ┗ 📜package.json
```
