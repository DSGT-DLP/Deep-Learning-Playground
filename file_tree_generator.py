import os
import regex as re

directories = ["backend", "frontend"]

# Content to ignore
IGNORED_DIRECTORIES = [
    ".next",
    "node_modules",
    ".venv",
    "__pycache__",
    "aws_rekognition_utils",
    "aws_secrets_utils",
    "lambda_utils",
    "s3_utils",
    "sqs_utils",
    "dlp_logging",
    "endpoints",
    "firebase_helpers",
    "tests",
    "tmp",
]
IGNORED_FILES = []

FILE_DESCRIPTIONS = {
    # Frontend file descriptions
    "settings.js": "List of layers and parameters supported in this playground",
    "iris.csv": "Sample CSV data",
    "index.js": "Calls the App.js file to render to the .root DOM element",
    "Home.js": "Main project file that renders the Home frontpage",
    "constants.js": "Constants for the frontend",
    "App.js": "Base React file",
    "App.css": "General CSS file",
    "demo_video.gif": "GIF tutorial of a simple classification training session",
    "dlp-logo.svg": "DLP Logo, duplicate of files in public, but essential as the frontend can't read public",
    "dlp-logo.png": "DLP Logo, duplicate of files in public, but essential as the frontend can't read public",
    "index.js": "Centralized location to import any components from outside of this ./components directory",
    "Wiki.js": "Primary Wiki page component",
    "softmax_equation.png": "Softmax equation screenshot for reference in Wiki",
    "Pretrained.js": "Primary Pretrained page component",
    "Transforms.js": "Renders a dropdown select and drag and drop component",
    "ImageModels.js": "Primary Image Models page component",
    "DataCodeSnippet.js": "Renders the dataloaders snippet",
    "TrainButton.js": "Renders the Train button, clicking which will call the backend with the ML/DL parameters",
    "Results.js": "Renders the results after a train session, or a simple text if no train sessions have been done or a session has failed with the backend's message",
    "RectContainer.js": "Renders a stylizable fixed-sized rectangle for the layers",
    "LayerChoice.js": "Renders a layer container for all the possible layers, for the user to drag from into the AddNewLayer component",
    "Input.js": "Renders the Parameters for the machine/deep learning tools (often DropDown input components)",
    "EmailInput.js": "Renders the email input form",
    "DropDown.js": "Renders the drop down components using react-select package",
    "CSVInputURL.js": "Renders the CSV URL input contents (if any)",
    "CSVInputFile.js": "Renders the CSV file input contents (if any)",
    "CodeSnippet.js": "Renders the code snippet container",
    "ChoiceTab.js": "Renders the navigation tab to switch between types of DL training",
    "BackgroundLayout.js": "Renders a light blue horizontally-stretched row to serve as a background to contain other components in one subsection in the Homepage",
    "AddNewLayer.js": "Renders a fillable layer container for the user to drag LayerChoice components into, where the new layer input will be filled by an AddedLayer component",
    "AddedLayer.js": "Renders an added layer container in the topmost row",
    "TrainButtonFunctions.js": "Stores the logic for validating and creating JSON to send to backend",
    "TalkWithBackend.js": "Sends ML/DL parameters to the backend and receives the backend",
    "TitleText.js": "Renders a simple header for a small subsection title",
    "LargeFileUpload.js": "Renders a dropzone component to upload large files",
    "Footer.js": "Primary Footer page component",
    "Footer.css": "CSS file for Footer",
    "Feedback.js": "Primary Feedback page component",
    "About.js": "Primary About page component",
    "my_deep_learning_model.onnx": "Last ONNX file output",
    "model.pt": "Last model.pt output",
    "manifest.json": "Default React file for choosing icon based on",
    "index.html": "Base HTML file that will be initially rendered",
    "dlp-logo.ico": "DLP Logo",
    "softmax_equation.png": "PNG file of Softmax equation",
    "Softmax.md": "Doc for Softmax layer",
    "ReLU.md": "Doc for ReLU later",
    "Linear.md": "Doc for Linear layer",
    # Backend file descriptions
    "app.py": "run the backend (entrypoint script)",
    "data.csv": "data csv file for use in the playground",
    "ml_trainer.py": "train a classical machine learning learning model on the dataset",
    "pretrained.py": "Functionality to support user training pretrained models (eg: alexnet, resnet, vgg16, etc) via timmodels + fast AI",
    "dl_trainer.py": "train a deep learning model on the dataset",
    "dl_model_parser.py": "parse the user specified pytorch model",
    "dl_model.py": "torch model based on user specifications from drag and drop",
    "dl_eval.py": "Evaluation functions for deep learning models in Pytorch (eg: accuracy, loss, etc)",
    "utils.py": "utility functions that could be helpful",
    "optimizer.py": "what optimizer to use (ie: SGD or Adam for now)",
    "loss_functions.py": "loss function enum",
    "email_notifier.py": "Endpoint to send email notification of training results via API Gateway + AWS SES",
    "default_datasets.py": "store logic to load in default datasets from scikit-learn",
    "dataset.py": "read in the dataset through URL or file upload",
    "constants.py": "list of helpful constants",
    "status_db.py": "General Dynamo DB table for status",
    "base_db.py": "General Dynamo DB Utility class that other Dynamo DB can inherit",
}


def traverse_directory(dir: str, is_root: bool, prefix: str) -> str:
    os.chdir(dir)
    output = ""
    if is_root:
        output = "📦 " + dir
    else:
        output = prefix + "📂 " + dir + ":"
    output += "\n"

    files = []
    subdirs = os.listdir()
    for subdir in subdirs:
        if os.path.isfile(subdir):
            files.append(subdir)
        elif subdir not in IGNORED_DIRECTORIES:
            subtree = traverse_directory(subdir, False, "|  " + prefix)
            output += subtree

    for file in files:
        skip = False
        for ignored_file in IGNORED_FILES:
            if re.match(ignored_file, file):
                skip = True
                break
        if skip:
            continue

        output += "|  " + prefix + "📜 " + file
        if file in FILE_DESCRIPTIONS:
            output += " : " + FILE_DESCRIPTIONS[file]
        output += "\n"

    os.chdir("../")
    return output


OUTPUT_FILE_DIRECTORY = ".github/Architecture.md"

content = "# Architecture\n\n"
for directory in directories:
    content += "## " + directory.capitalize() + " Architecture\n\n"
    content += "```\n"
    content += traverse_directory(directory, True, "|- ")
    content += "```"
    content += "\n\n"

f = open(OUTPUT_FILE_DIRECTORY, "w", encoding="utf-8")
f.write(content)
f.close()
