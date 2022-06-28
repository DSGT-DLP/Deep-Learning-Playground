# Architecture

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

## Frontend Architecture

```
📦playground-frontend
 ┣ 📂layer_docs
 ┃ ┣ 📜Linear.md - Doc for Linear layer 
 ┃ ┣ 📜ReLU.md - Doc for ReLU later
 ┃ ┣ 📜Softmax.md - Doc for Softmax layer
 ┃ ┗ 📜softmax_equation.png - PNG file of Softmax equation
 ┣ 📂public
 ┃ ┣ 📜favicon.ico - Default React icon
 ┃ ┣ 📜index.html - Base HTML file that will be initially rendered
 ┃ ┣ 📜logo192.png - Default React icon for tab logo
 ┃ ┣ 📜manifest.json - Default React file for choosing icon based on preferred size
 ┃ ┗ 📜robots.txt - Default React file
 ┣ 📂src
 ┃ ┣ 📂backend_outputs
 ┃ ┃ ┣ 📂visualization_output
 ┃ ┃ ┃ ┗ 📜my_confusion_matrix.png - Last Confusion Matrix output file
 ┃ ┃ ┣ 📜model.pt - Last .pt output file
 ┃ ┃ ┗ 📜my_deep_learning_model.onnx - Last .onnx output file
 ┃ ┣ 📂components
 ┃ ┃ ┣ 📂About
 ┃ ┃ ┃ ┗ 📜About.js - Renders the About page
 ┃ ┃ ┣ 📂Footer
 ┃ ┃ ┃ ┣ 📜Footer.css - CSS file for Footer
 ┃ ┃ ┃ ┗ 📜Footer.js - Renders the Footer
 ┃ ┃ ┣ 📂mini_components
 ┃ ┃ ┃ ┗ 📜TitleText.js - Renders a simple header for a small subsection title
 ┃ ┃ ┣ 📂Navbar
 ┃ ┃ ┃ ┗ 📜Navbar.js - Renders the Navbar
 ┃ ┃ ┣ 📂Wiki
 ┃ ┃ ┃ ┣ 📜Wiki.js - Renders the Wiki page
 ┃ ┃ ┣ 📜AddedLayer.js - Renders a added layer container in the topmost row
 ┃ ┃ ┣ 📜AddNewLayer.js - Renders a fillable layer container for the user to drag LayerChoice components into, where the new layer input will be filled by an AddedLayer component 
 ┃ ┃ ┣ 📜BackgroundLayout.js - Renders a light blue horizontally-stretched row to serve as a background to contain other components in one subsection in the Homepage
 ┃ ┃ ┣ 📜CodeSnippet.js - Renders the code snippet container
 ┃ ┃ ┣ 📜CSVInput.js - Renders the CSV file input contents(if any)
 ┃ ┃ ┣ 📜DropDown.js - Renders the drop down components using react-select package
 ┃ ┃ ┣ 📜EmailInput.js - Renders the email input form
 ┃ ┃ ┣ 📜FeedBackForm.js - Renders the Feedback Form page
 ┃ ┃ ┣ 📜index.js - Centralized location to import any components from outside of this ./components directory
 ┃ ┃ ┣ 📜Input.js - Renders the Parameters for the machine/deep learning tools (often DropDown input components)
 ┃ ┃ ┣ 📜LayerChoice.js - Renders a layer container for all the possible layers, for the user to drag from into the AddNewLayer component
 ┃ ┃ ┣ 📜RectContainer.js - Renders a stylizable fixed-sized rectangle for the layers
 ┃ ┃ ┗ 📜TrainButton.js - Renders the Train button, clicking which will call the backend with the ML/DL parameters
 ┃ ┣ 📂helper_functions
 ┃ ┃ ┗ 📜TalkWithBackend.js - Sends ML/DL parameters to the backend and receives the backend response, setting the frontend states triggering an output display change
 ┃ ┣ 📂images
 ┃ ┃ ┣ 📂logos
 ┃ ┃ ┃ ┣ 📜aws-logo.png
 ┃ ┃ ┃ ┣ 📜dsgt-logo-dark.png
 ┃ ┃ ┃ ┣ 📜dsgt-logo-light.png
 ┃ ┃ ┃ ┣ 📜dsgt-logo-white-back.png
 ┃ ┃ ┃ ┣ 📜flask-logo.png
 ┃ ┃ ┃ ┣ 📜pandas-logo.svg
 ┃ ┃ ┃ ┣ 📜python-logo.png
 ┃ ┃ ┃ ┣ 📜pytorch-logo.png
 ┃ ┃ ┃ ┗ 📜react-logo.png
 ┃ ┃ ┗ 📜demo_video.gif
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
