# Architecture

## Training Architecture

```
📦 training
|  |- 📂 training:
|  |  |- 📂 routes:
|  |  |  |- 📂 tabular:
|  |  |  |  |- 📜 __init__.py
|  |  |  |  |- 📜 schemas.py
|  |  |  |  |- 📜 tabular.py
|  |  |  |- 📂 datasets:
|  |  |  |  |- 📂 default:
|  |  |  |  |  |- 📜 columns.py
|  |  |  |  |  |- 📜 __init__.py
|  |  |  |  |  |- 📜 schemas.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |- 📂 image:
|  |  |  |  |- 📜 image.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |  |- 📜 schemas.py
|  |  |  |- 📜 __init__.py
|  |  |  |- 📜 schemas.py
|  |  |- 📂 core:
|  |  |  |- 📜 dl_model.py : torch model based on user specifications from drag and drop
|  |  |  |- 📜 trainer.py
|  |  |  |- 📜 authenticator.py
|  |  |  |- 📜 __init__.py
|  |  |  |- 📜 dataset.py : read in the dataset through URL or file upload
|  |  |  |- 📜 criterion.py
|  |  |  |- 📜 optimizer.py : what optimizer to use (ie: SGD or Adam for now)
|  |  |- 📜 settings.py
|  |  |- 📜 urls.py
|  |  |- 📜 wsgi.py
|  |  |- 📜 asgi.py
|  |  |- 📜 __init__.py
|  |- 📜 poetry.lock
|  |- 📜 pyproject.toml
|  |- 📜 docker-compose.prod.yml
|  |- 📜 docker-compose.yml
|  |- 📜 Dockerfile.prod
|  |- 📜 pytest.ini
|  |- 📜 manage.py
|  |- 📜 environment.yml
|  |- 📜 README.md
|  |- 📜 Dockerfile
|  |- 📜 cli.py
```

## Frontend Architecture

```
📦 frontend
|  |- 📂 layer_docs:
|  |  |- 📜 Linear.md : Doc for Linear layer
|  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |- 📜 Softmax.md : Doc for Softmax layer
|  |  |- 📜 ReLU.md : Doc for ReLU later
|  |- 📂 public:
|  |  |- 📂 images:
|  |  |  |- 📂 learn_mod_images:
|  |  |  |  |- 📜 binarystepactivation.png
|  |  |  |  |- 📜 lossExampleEquation.png
|  |  |  |  |- 📜 robotImage.jpg
|  |  |  |  |- 📜 sigmoidfunction.png
|  |  |  |  |- 📜 tanhactivation.png
|  |  |  |  |- 📜 lossExampleTable.png
|  |  |  |  |- 📜 neuron.png
|  |  |  |  |- 📜 sigmoidactivation.png
|  |  |  |  |- 📜 ReLUactivation.png
|  |  |  |  |- 📜 lossExample.png
|  |  |  |  |- 📜 neuralnet.png
|  |  |  |  |- 📜 neuronWithEquation.png
|  |  |  |  |- 📜 LeakyReLUactivation.png
|  |  |  |- 📂 wiki_images:
|  |  |  |  |- 📜 tanh_equation.png
|  |  |  |  |- 📜 avgpool_maxpool.gif
|  |  |  |  |- 📜 batchnorm_diagram.png
|  |  |  |  |- 📜 conv2d.gif
|  |  |  |  |- 📜 tanh_plot.png
|  |  |  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |  |  |- 📜 dropout_diagram.png
|  |  |  |  |- 📜 sigmoid_equation.png
|  |  |  |  |- 📜 conv2d2.gif
|  |  |  |  |- 📜 maxpool2d.gif
|  |  |  |- 📂 logos:
|  |  |  |  |- 📂 dlp_branding:
|  |  |  |  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |- 📜 flask-logo.png
|  |  |  |  |- 📜 dsgt-logo-white-back.png
|  |  |  |  |- 📜 aws-logo.png
|  |  |  |  |- 📜 pandas-logo.png
|  |  |  |  |- 📜 react-logo.png
|  |  |  |  |- 📜 github.png
|  |  |  |  |- 📜 google.png
|  |  |  |  |- 📜 dsgt-logo-dark.png
|  |  |  |  |- 📜 dsgt-logo-light.png
|  |  |  |  |- 📜 python-logo.png
|  |  |  |  |- 📜 pytorch-logo.png
|  |  |  |- 📜 demo_video.gif : GIF tutorial of a simple classification training session
|  |  |- 📜 dlp-logo.ico : DLP Logo
|  |  |- 📜 index.html : Base HTML file that will be initially rendered
|  |  |- 📜 robots.txt
|  |  |- 📜 manifest.json : Default React file for choosing icon based on
|  |- 📂 src:
|  |  |- 📂 features:
|  |  |  |- 📂 OpenAi:
|  |  |  |  |- 📜 openAiUtils.ts
|  |  |  |- 📂 Feedback:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 feedbackApi.ts
|  |  |  |- 📂 Train:
|  |  |  |  |- 📂 types:
|  |  |  |  |  |- 📜 trainTypes.ts
|  |  |  |  |- 📂 features:
|  |  |  |  |  |- 📂 Image:
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 imageTypes.ts
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 imageApi.ts
|  |  |  |  |  |  |  |- 📜 imageActions.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 ImageParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageFlow.tsx
|  |  |  |  |  |  |  |- 📜 ImageDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageTrainspace.tsx
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 imageConstants.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |  |- 📂 Tabular:
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 tabularTypes.ts
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 tabularApi.ts
|  |  |  |  |  |  |  |- 📜 tabularActions.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 TabularDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 TabularFlow.tsx
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 tabularConstants.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 trainspaceApi.ts
|  |  |  |  |  |- 📜 trainspaceSlice.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 CreateTrainspace.tsx
|  |  |  |  |  |- 📜 DatasetStepLayout.tsx
|  |  |  |  |  |- 📜 TrainspaceLayout.tsx
|  |  |  |  |- 📂 constants:
|  |  |  |  |  |- 📜 trainConstants.ts
|  |  |  |- 📂 LearnMod:
|  |  |  |  |- 📜 Exercise.tsx
|  |  |  |  |- 📜 MCQuestion.tsx
|  |  |  |  |- 📜 FRQuestion.tsx
|  |  |  |  |- 📜 ClassCard.tsx
|  |  |  |  |- 📜 LearningModulesContent.tsx
|  |  |  |  |- 📜 ImageComponent.tsx
|  |  |  |  |- 📜 ModulesSideBar.tsx
|  |  |  |- 📂 Dashboard:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 dashboardApi.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TrainDoughnutChart.tsx
|  |  |  |  |  |- 📜 TrainBarChart.tsx
|  |  |  |  |  |- 📜 TrainDataGrid.tsx
|  |  |- 📂 pages:
|  |  |  |- 📂 train:
|  |  |  |  |- 📜 [train_space_id].tsx
|  |  |  |  |- 📜 index.tsx
|  |  |  |- 📜 about.tsx
|  |  |  |- 📜 dashboard.tsx
|  |  |  |- 📜 _document.tsx
|  |  |  |- 📜 learn.tsx
|  |  |  |- 📜 LearnContent.tsx
|  |  |  |- 📜 wiki.tsx
|  |  |  |- 📜 forgot.tsx
|  |  |  |- 📜 settings.tsx
|  |  |  |- 📜 _app.tsx
|  |  |  |- 📜 feedback.tsx
|  |  |  |- 📜 login.tsx
|  |  |- 📂 backend_outputs:
|  |  |  |- 📜 model.pkl
|  |  |  |- 📜 model.pt : Last model.pt output
|  |  |  |- 📜 my_deep_learning_model.onnx : Last ONNX file output
|  |  |- 📂 common:
|  |  |  |- 📂 redux:
|  |  |  |  |- 📜 store.ts
|  |  |  |  |- 📜 train.ts
|  |  |  |  |- 📜 backendApi.ts
|  |  |  |  |- 📜 hooks.ts
|  |  |  |  |- 📜 userLogin.ts
|  |  |  |- 📂 components:
|  |  |  |  |- 📜 EmailInput.tsx
|  |  |  |  |- 📜 TitleText.tsx
|  |  |  |  |- 📜 NavBarMain.tsx
|  |  |  |  |- 📜 Spacer.tsx
|  |  |  |  |- 📜 DlpTooltip.tsx
|  |  |  |  |- 📜 Footer.tsx
|  |  |  |  |- 📜 HtmlTooltip.tsx
|  |  |  |  |- 📜 ClientOnlyPortal.tsx
|  |  |  |- 📂 utils:
|  |  |  |  |- 📜 dateFormat.ts
|  |  |  |  |- 📜 dndHelpers.ts
|  |  |  |  |- 📜 firebase.ts
|  |  |  |- 📂 styles:
|  |  |  |  |- 📜 globals.css
|  |  |  |  |- 📜 Home.module.css
|  |  |- 📜 GlobalStyle.ts
|  |  |- 📜 constants.ts
|  |  |- 📜 iris.csv : Sample CSV data
|  |  |- 📜 next-env.d.ts
|  |- 📜 .eslintrc.json
|  |- 📜 yarn.lock
|  |- 📜 package.json
|  |- 📜 .eslintignore
|  |- 📜 next.config.js
|  |- 📜 pnpm-lock.yaml
|  |- 📜 tsconfig.json
|  |- 📜 next-env.d.ts
|  |- 📜 jest.config.js
```

