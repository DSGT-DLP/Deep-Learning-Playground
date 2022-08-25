# Deep Learning Playground

Web Application where people new to Deep Learning can input a dataset and toy around with basic Pytorch modules through a drag and drop interface.

> **Deployed website:** https://datasciencegt-dlp.com/ </br>
 **GitHub repo:** https://github.com/karkir0003/Deep-Learning-Playground/ </br> **Owners:** See [CODEOWNERS](./CODEOWNERS)

# How to Run

## Prerequisites
Have the following installed first:

1. [NodeJS v16](https://nodejs.org/en/download/)
2. NPM v8
3. [Anaconda](https://www.anaconda.com/)

## Shortcuts:
| Terminal   | Procedure   | Command |
|-----------|-----------|-----|
| Command Prompt | Run All | `start /B npm run startf & start npm run startb` |
| Powershell | Run All | `cmd /C 'start /B npm run startf & start npm run startb'` |
| Linux Terminal | Run All | `tmux new-session -d -s frontend_session 'npm run startf'; tmux new-session -d -s backend_session 'npm run startb'` |


## Installing the frontend (one-time)
In the root directory of this project (`~/Deep-Learning-Playground`), run: 
```
npm run installf
```
## Installing the backend (one-time)
```
npm run installb
```
## Running the backend
```
npm run startb
```
## Running the frontend
```
npm run startf
```

# Backend

## Conda Env Setup

- `conda env create -f environment.yml` in the `/conda` directory

- Updating an environment: `conda env update -f environment.yml` in the `/conda` directory

## Backend Infra

`python -m backend.driver` from the `~/Deep-Learning-Playground` directory

The backend supports training of a deep learning model and/or a classical ML model!

## Backend Architecture

See [Architecture.md](./.github/Architecture.md)

## Examples

To see how `driver.py` is used, see [`Backend_Examples.md`](./.github/Backend_Examples.md)

# Frontend

## Startup Instructions

> **Note:** You will need the `.env` file from @farisdurrani to get the `Feedback` page working, but other pages work fine without it

1. For complete functionality with the backend, first, start the backend using the instructions above. The backend will be live at http://localhost:8000/

2. Then in a separate terminal, start the frontend development server. After installing [nodeJS v16](https://nodejs.org/en/download/), run the following commands:

```
cd frontend\playground-frontend
npm install
npm start
```

3. Then, go to http://localhost:3000/

## Frontend Architecture

See [Architecture.md](./.github/Architecture.md)

# License

Deep Learning Playground is MIT licensed, as found in the [LICENSE](./LICENSE) file.

Deep Learning Playground documentation is Creative Commons licensed, as found in the [LICENSE-docs](./.github/LICENSE-docs) file.
