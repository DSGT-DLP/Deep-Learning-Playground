# Deep Learning Playground

Web Application where people new to Machine Learning can input a dataset and experiment around with basic Pytorch modules through a drag and drop interface

> **Deployed website:** https://datasciencegt-dlp.com/ </br> > **GitHub repo:** https://github.com/DSGT-DLP/Deep-Learning-Playground </br> > **Owners:** See [CODEOWNERS](./CODEOWNERS)

# How to Run

## Prerequisites

Have the following installed first:

1. [NodeJS v18](https://nodejs.org/en/download/) (should come with NPM v9, you must install Yarn v1.22 afterwards using NPM)
1. [Poetry](https://python-poetry.org/)
1. [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html). After installing, type `aws configure` in your terminal and type in the credentials given in [Secrets](https://docs.google.com/spreadsheets/d/1fRndo-7u0MXghiZoMp3uBepDBW9EghcJ9IL4yS0TdD8/edit?usp=sharing)

## Recommended

1. [GitKraken](https://help.gitkraken.com/gitkraken-client/how-to-install/) for helping with Git commands, especially if you're new to Git
1. [Node Version Manager](https://www.freecodecamp.org/news/node-version-manager-nvm-install-guide/) for managing NodeJS versions
1. [pyenv](https://github.com/pyenv/pyenv) for managing python versions

## To start on localhost (in order):

| Action                                        | Command                        |
| --------------------------------------------- | ------------------------------ |
| Install (one-time) / Update Frontend Packages | `yarn run installf`            |
| Install Backend Packages (one-time)           | `yarn run installb`            |
| Update Backend Packages                       | `cd backend && poetry install` |
| Get secrets                                   | `yarn run secrets`             |
| Running the Frontend                          | `yarn run startf`              |
| Running the Backend                           | `yarn run startb`              |

## To run in `production` mode:

- To start the frontend:
  - If you're using Unix, run `REACT_APP_MODE=prod yarn run startf` from the root of the project
  - If you're using Windows, run `set REACT_APP_MODE=prod && yarn run startf` from the root of the project
- Run the backend as usual
- To run the SQS container, run the command in this [document](https://docs.google.com/document/d/1yYzT7CCUqxnShncHEeHC1MssABntJuKUN88pTXnh_HQ/edit#)

# Development Practices

## GitGuardian Pre-commit Check

To protect our secrets, we use the GitGuardian ggshield pre-commit check to ensure no keys are being committed. After installing the backend, run

```sh
pre-commit install
```

You should get output like "pre-commit installed at .git/hooks/pre-commit". Login to GitGuardian to activate the pre-commit hook using

```sh
ggshield auth login
```

If this command works properly, you will be redirected to an auth route in the Git Guardian website. **Sign in using your Github account**. Then, you should be all set!

# Further Details: Backend

## Poetry Env Setup

- `poetry install` in the project root directory (this installs both dev and prod dependencies). Make sure you run `pip install poetry` prior to running this command

- Updating dependencies: `poetry update`

- If you encounter any error related to "ChefBuildError", downgrade your poetry version to 1.3.2 by running `pip install poetry==1.3.2` before doing `poetry install` (See related github issue [here](https://github.com/python-poetry/poetry/issues/7611))

## pyenv setup

### Mac Instructions

- To install pyenv, a python version management tool, you can use the following command via homebrew: `brew install pyenv`

- Installing python version: `pyenv install 3.9.17`

- Set the global python version: `pyenv global 3.9.17`

- Verify the installation using `pyenv --version`

- If you encounter any issues related to Python versions or missing import modules (no modules named "x"), you can solve by:
  `export CONFIGURE_OPTS="--with-openssl=$(brew --prefix openssl)"`
  `pyenv install -v 3.9.17`

### Windows instructions

- Open up Windows Powershell as Administrator
- Follow the setup instructions for pyenv [here](https://github.com/pyenv-win/pyenv-win/blob/master/docs/installation.md#powershell)
- Run `pyenv install 3.9.13`
- Set global version by running `pyenv global 3.9.13`

## Backend Infrastructure

`poetry run python app.py` from the `~/Deep-Learning-Playground/backend` directory

The backend supports training of a deep learning model and/or a classical ML model

## Backend Architecture

See [Architecture.md](./.github/Architecture.md)

## Examples

To see how `app.py` is used, see [`Backend_Examples.md`](./.github/Backend_Examples.md)

# Further Details: Frontend

## Startup Instructions

> **Note:** You will need the `.env` file to get certain pages working. See the command `Get secrets` above to get the `.env` file.

1. For complete functionality with the backend, first, start the backend using the instructions above. The backend will be live at http://localhost:8000/

2. Then in a separate terminal, start the frontend development server. After installing the prerequisites above, run the shortcut above or run the following commands:

   ```
   cd frontend
   yarn install
   yarn start
   ```

3. Then, go to http://localhost:3000/

# Tmux Instructions

## Windows Installation

1. [Install WSL](https://code.visualstudio.com/docs/remote/wsl) and make sure you get the Ubuntu distro

2. Open the Ubuntu terminal

3. [Install NodeJS](https://www.digitalocean.com/community/tutorials/how-to-install-node-js-on-ubuntu-20-04)

4. Installing AWS:
   Type `sudo apt install unzip`.
   Follow these [instructions](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).
   Type `aws configure` and enter secrets.

5. Navigate to project directory.
   Type `yarn run installf`.
   If your NodeJS is outdated, follow these [instructions](https://www.hostingadvice.com/how-to/update-node-js-latest-version/).

6. Enter these commands:

```
yarn run installb
yarn run secrets
```

## Mac Installation

1. Type `brew install tmux`

## Running the script

1. Type `./tmux-script.sh`
2. Visit [cheatsheet website](https://gist.github.com/MohamedAlaa/2961058)

## Frontend Architecture

See [Architecture.md](./.github/Architecture.md)

# License

Deep Learning Playground is MIT licensed, as found in the [LICENSE](./LICENSE) file.

Deep Learning Playground documentation is Creative Commons licensed, as found in the [LICENSE-docs](./.github/LICENSE-docs) file.
