# Deep-Learning-Playground
Web Application where people new to Deep Learning can input a dataset and toy around with basic Pytorch modules through a drag and drop interface.

> **Deployed website:** https://datasciencegt-dlp.com/ </br>
> **GitHub repo:** https://github.com/karkir0003/Deep-Learning-Playground/ </br>
> **Owners:** See [CODEOWNERS](./CODEOWNERS)

# Backend

## Conda Env Setup
* `conda env create -f environment.yml` in the `/conda` directory

* Updating an environment: `conda env update -f environment.yml` in the `/conda` directory
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

1. For complete functionality with the backend, first, start the backend using the instructions above. The backend will be live at http://localhost:5000/run

2. Then in a separate terminal, start the frontend development server. After installing [nodeJS v16](https://nodejs.org/en/download/), run the following commands:
```
cd frontend\playground-frontend
npm install
npm start
```
3. Then, go to http://localhost:3000/

## Frontend Architecture
See [Architecture.md](./.github/Architecture.md)

## How to Add New Layer Options
Currently, there are three layers implemented in this playgroud—Linear, ReLU, and Softmax. A developer can easily add in a new layer to be used by the user through:
1. Go to [settings.js](./frontend/playground-frontend/src/settings.js)
2. Put in (* = required):
    - `display_name`*: Name of layer to be displayed to user
    - `object_name`*: Layer object to be passed into the backend, e.g., `nn.linear(...)`
    - `parameters`: An array of JS objects with at least the display name of the parameters for the layer object

# License

Deep Learning Playground is MIT licensed, as found in the [LICENSE](./LICENSE) file.

Deep Learning Playground documentation is Creative Commons licensed, as found in the [LICENSE-docs](./.github/LICENSE-docs) file.
