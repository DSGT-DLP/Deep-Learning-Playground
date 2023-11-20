# Deep Learning Playground

Web Application where people new to Machine Learning can input a dataset and experiment around with basic Pytorch modules through a drag and drop interface

> **Deployed website:** https://datasciencegt-dlp.com/ </br> > **GitHub repo:** https://github.com/DSGT-DLP/Deep-Learning-Playground

# Getting Started

### Prerequisites

Have the following installed first:

1. [Node.js v18 via NVM](https://github.com/nvm-sh/nvm#installing-and-updating) (Install nvm first, and then install node & npm using nvm)
1. [Yarn 1.x](https://classic.yarnpkg.com/lang/en/docs/install) (Must be installed after npm. May upgrade to Yarn Modern in the future, keep an eye out for that!)
1. [Mamba](https://github.com/conda-forge/miniforge#miniforge) (Make sure to install using the Miniforge distribution. On windows, remember to check the box that says that it will add mamba to path)
1. [pip](https://pip.pypa.io/en/stable/installation/) (Is also automatically installed with Python via Python's installer, make sure this version of pip is installed globally)
1. [dlp-cli](https://github.com/DSGT-DLP/dlp-cli#readme) (We have our own cli!)
1. [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
1. [VSCode](https://code.visualstudio.com/)
1. [git](https://git-scm.com/downloads)

### Recommended

1. [GitKraken](https://help.gitkraken.com/gitkraken-client/how-to-install/) for helping with Git commands, especially if you're new to Git
1. [Postman](https://www.postman.com/downloads/) (Extremely helpful for testing REST APIs)
1. [Chrome](https://www.google.com/chrome/) (For Chrome developer tools)
1. [Redux Devtools](https://chrome.google.com/webstore/detail/redux-devtools/lmhkpmbekcpmknklioeibfkpmmfibljd) (Helpful for debugging any Redux)
1. [Docker](https://www.docker.com/)
1. [go](https://go.dev/doc/install) (In case if you ever need to contribute to the dlp-cli)
1. VSCode Extensions:
   1. [Github Copilot](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot)
   1. [IntelliCode](https://marketplace.visualstudio.com/items?itemName=VisualStudioExptTeam.vscodeintellicode)
   1. [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
   1. [Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
   1. [Pretter - Code formatter](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode)
   1. [ESLint](https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint)
   1. [Docker](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker)
   1. [Go](https://marketplace.visualstudio.com/items?itemName=golang.Go)

## Clone the Repository

`git clone https://github.com/DSGT-DLP/Deep-Learning-Playground.git` in the directory of your choice. If this installation is for the beginner project, do `git clone https://github.com/DSGT-DLP/dlp-practice.git`.

This can also be achieved through GitKraken.

## Frontend and Backend Package Installation

Run the following commands in the project directory (the root folder created after cloning):

| Action                           | Command                    |
| -------------------------------- | -------------------------- |
| Install/Update Frontend Packages | `dlp-cli frontend install` |
| Install/Update Backend Packages  | `dlp-cli backend install`  |

## GitGuardian Pre-commit Check

To install the GitGuardian cli and pre-commit, run

```sh
pip install ggshield
pip install pre-commit
```

To protect our secrets, we use the GitGuardian ggshield pre-commit check to ensure no keys are being committed. After installing the backend, run

```sh
pre-commit install
```

You should get output like "pre-commit installed at .git/hooks/pre-commit". Login to GitGuardian to activate the pre-commit hook using

```sh
ggshield auth login
```

If this command works properly, you will be redirected to an auth route in the Git Guardian website. **Sign in using your Github account**. Then, you should be all set!

## Additional VSCode Setup (Recommended)

Access the VSCode command palette via `Ctrl+Shift+P`. Press `Python: Select Interpreter`. You need the Python VSCode extension for this.

Select the Python Interpreter named `dlp`.

## To start on localhost:

Run the following commands in the project directory (the root folder created after cloning):

| Action               | Command                  |
| -------------------- | ------------------------ |
| Running the Frontend | `dlp-cli frontend start` |
| Running the Backend  | `dlp-cli backend start`  |

Make sure to run the above two commands in separate terminals.

## AWS Setup
If you will be working on tasks that interface with AWS resources/services, please follow the below steps (please install AWS CLI using this [link](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) first):

1. Request an AWS Account for Deep Learning Playground by messaging Faris, Karthik, or Daniel in the DLP Discord. Please include your Github username along with your personal email account
1. Once an AWS Account has been created, you will receive an email from AWS that will require you to setup a password
1. When you login, you should be seeing that the account you're added under is `Data Science Initiative Inc`
1. Click on the dropdown to expand the `Data Science Initiative Inc` entry and select the `Command Line or programmatic access button`
1. Open your terminal and navigate to the DLP directory
   1. Run `aws configure sso``. Follow the prompts to enter the SSO Start URL (this comes from step 2) and the below values
   ```
   sso_region = us-east-1
   sso_session = dlp
   sso_registration_scopes = sso:account:access
   default output format = None
   cli profile name = just press enter (use the default one provided)
   ````
   1. Make sure you follow the instructions in the terminal to ensure your credentials are set correctly (eg: allow botocore to access data should be selected as "yes")
   1. Run `cat ~/.aws/config` to look for the sso profile configured.
   1. Run `export AWS_PROFILE=<sso_profile_name from step 6>`

Please message in the DLP Discord if you have any difficulty/issue with these steps. 

# Architecture

See [Architecture.md](./.github/Architecture.md)

# License

Deep Learning Playground is MIT licensed, as found in the [LICENSE](./LICENSE) file.

Deep Learning Playground documentation is Creative Commons licensed, as found in the [LICENSE-docs](./.github/LICENSE-docs) file.
