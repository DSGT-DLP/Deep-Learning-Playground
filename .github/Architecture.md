# Architecture

## Training Architecture

```
📦 training
|  |- 📂 training:
|  |  |- 📂 routes:
|  |  |  |- 📂 datasets:
|  |  |  |  |- 📂 default:
|  |  |  |  |  |- 📜 columns.py
|  |  |  |  |  |- 📜 __init__.py
|  |  |  |  |  |- 📜 schemas.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |- 📂 tabular:
|  |  |  |  |- 📜 __init__.py
|  |  |  |  |- 📜 tabular.py
|  |  |  |  |- 📜 schemas.py
|  |  |  |- 📂 image:
|  |  |  |  |- 📜 image.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |  |- 📜 schemas.py
|  |  |  |- 📜 __init__.py
|  |  |  |- 📜 schemas.py
|  |  |- 📂 core:
|  |  |  |- 📜 dataset.py : read in the dataset through URL or file upload
|  |  |  |- 📜 __init__.py
|  |  |  |- 📜 authenticator.py
|  |  |  |- 📜 optimizer.py : what optimizer to use (ie: SGD or Adam for now)
|  |  |  |- 📜 trainer.py
|  |  |  |- 📜 dl_model.py : torch model based on user specifications from drag and drop
|  |  |  |- 📜 criterion.py
|  |  |- 📜 settings.py
|  |  |- 📜 asgi.py
|  |  |- 📜 wsgi.py
|  |  |- 📜 __init__.py
|  |  |- 📜 urls.py
|  |- 📜 environment.yml
|  |- 📜 poetry.lock
|  |- 📜 manage.py
|  |- 📜 docker-compose.yml
|  |- 📜 cli.py
|  |- 📜 docker-compose.prod.yml
|  |- 📜 pytest.ini
|  |- 📜 pyproject.toml
|  |- 📜 README.md
|  |- 📜 Dockerfile
```

## Frontend Architecture

```
📦 frontend
|  |- 📂 layer_docs:
|  |  |- 📜 ReLU.md : Doc for ReLU later
|  |  |- 📜 Linear.md : Doc for Linear layer
|  |  |- 📜 Softmax.md : Doc for Softmax layer
|  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |- 📂 public:
|  |  |- 📂 images:
|  |  |  |- 📂 logos:
|  |  |  |  |- 📂 dlp_branding:
|  |  |  |  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |- 📜 flask-logo.png
|  |  |  |  |- 📜 dsgt-logo-dark.png
|  |  |  |  |- 📜 dsgt-logo-light.png
|  |  |  |  |- 📜 python-logo.png
|  |  |  |  |- 📜 github.png
|  |  |  |  |- 📜 google.png
|  |  |  |  |- 📜 aws-logo.png
|  |  |  |  |- 📜 react-logo.png
|  |  |  |  |- 📜 pandas-logo.png
|  |  |  |  |- 📜 pytorch-logo.png
|  |  |  |  |- 📜 dsgt-logo-white-back.png
|  |  |  |- 📂 wiki_images:
|  |  |  |  |- 📜 avgpool_maxpool.gif
|  |  |  |  |- 📜 dropout_diagram.png
|  |  |  |  |- 📜 tanh_equation.png
|  |  |  |  |- 📜 maxpool2d.gif
|  |  |  |  |- 📜 conv2d2.gif
|  |  |  |  |- 📜 sigmoid_equation.png
|  |  |  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |  |  |- 📜 conv2d.gif
|  |  |  |  |- 📜 batchnorm_diagram.png
|  |  |  |  |- 📜 tanh_plot.png
|  |  |  |- 📂 learn_mod_images:
|  |  |  |  |- 📜 sigmoidactivation.png
|  |  |  |  |- 📜 sigmoidfunction.png
|  |  |  |  |- 📜 lossExampleTable.png
|  |  |  |  |- 📜 lossExample.png
|  |  |  |  |- 📜 LeakyReLUactivation.png
|  |  |  |  |- 📜 robotImage.jpg
|  |  |  |  |- 📜 tanhactivation.png
|  |  |  |  |- 📜 ReLUactivation.png
|  |  |  |  |- 📜 neuron.png
|  |  |  |  |- 📜 binarystepactivation.png
|  |  |  |  |- 📜 lossExampleEquation.png
|  |  |  |  |- 📜 neuronWithEquation.png
|  |  |  |  |- 📜 neuralnet.png
|  |  |  |- 📜 demo_video.gif : GIF tutorial of a simple classification training session
|  |  |- 📜 manifest.json : Default React file for choosing icon based on
|  |  |- 📜 robots.txt
|  |  |- 📜 index.html : Base HTML file that will be initially rendered
|  |  |- 📜 dlp-logo.ico : DLP Logo
|  |- 📂 src:
|  |  |- 📂 features:
|  |  |  |- 📂 LearnMod:
|  |  |  |  |- 📜 LearningModulesContent.tsx
|  |  |  |  |- 📜 FRQuestion.tsx
|  |  |  |  |- 📜 MCQuestion.tsx
|  |  |  |  |- 📜 Exercise.tsx
|  |  |  |  |- 📜 ModulesSideBar.tsx
|  |  |  |  |- 📜 ImageComponent.tsx
|  |  |  |  |- 📜 ClassCard.tsx
|  |  |  |- 📂 Dashboard:
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TrainBarChart.tsx
|  |  |  |  |  |- 📜 TrainDoughnutChart.tsx
|  |  |  |  |  |- 📜 TrainDataGrid.tsx
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 dashboardApi.ts
|  |  |  |- 📂 OpenAi:
|  |  |  |  |- 📜 openAiUtils.ts
|  |  |  |- 📂 Feedback:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 feedbackApi.ts
|  |  |  |- 📂 Train:
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 CreateTrainspace.tsx
|  |  |  |  |  |- 📜 TrainspaceLayout.tsx
|  |  |  |  |  |- 📜 DatasetStepLayout.tsx
|  |  |  |  |- 📂 features:
|  |  |  |  |  |- 📂 Image:
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 ImageFlow.tsx
|  |  |  |  |  |  |  |- 📜 ImageDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageTrainspace.tsx
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 imageTypes.ts
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 imageConstants.ts
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 imageApi.ts
|  |  |  |  |  |  |  |- 📜 imageActions.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |  |- 📂 Tabular:
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 TabularTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 TabularDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularFlow.tsx
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 tabularTypes.ts
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 tabularConstants.ts
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 tabularApi.ts
|  |  |  |  |  |  |  |- 📜 tabularActions.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |- 📂 types:
|  |  |  |  |  |- 📜 trainTypes.ts
|  |  |  |  |- 📂 constants:
|  |  |  |  |  |- 📜 trainConstants.ts
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 trainspaceApi.ts
|  |  |  |  |  |- 📜 trainspaceSlice.ts
|  |  |- 📂 pages:
|  |  |  |- 📂 train:
|  |  |  |  |- 📜 [train_space_id].tsx
|  |  |  |  |- 📜 index.tsx
|  |  |  |- 📜 forgot.tsx
|  |  |  |- 📜 LearnContent.tsx
|  |  |  |- 📜 _document.tsx
|  |  |  |- 📜 settings.tsx
|  |  |  |- 📜 feedback.tsx
|  |  |  |- 📜 wiki.tsx
|  |  |  |- 📜 _app.tsx
|  |  |  |- 📜 about.tsx
|  |  |  |- 📜 login.tsx
|  |  |  |- 📜 dashboard.tsx
|  |  |  |- 📜 learn.tsx
|  |  |- 📂 backend_outputs:
|  |  |  |- 📜 model.pt : Last model.pt output
|  |  |  |- 📜 model.pkl
|  |  |  |- 📜 my_deep_learning_model.onnx : Last ONNX file output
|  |  |- 📂 common:
|  |  |  |- 📂 components:
|  |  |  |  |- 📜 NavBarMain.tsx
|  |  |  |  |- 📜 ClientOnlyPortal.tsx
|  |  |  |  |- 📜 HtmlTooltip.tsx
|  |  |  |  |- 📜 Spacer.tsx
|  |  |  |  |- 📜 EmailInput.tsx
|  |  |  |  |- 📜 Footer.tsx
|  |  |  |  |- 📜 DlpTooltip.tsx
|  |  |  |  |- 📜 TitleText.tsx
|  |  |  |- 📂 styles:
|  |  |  |  |- 📜 globals.css
|  |  |  |  |- 📜 Home.module.css
|  |  |  |- 📂 utils:
|  |  |  |  |- 📜 dndHelpers.ts
|  |  |  |  |- 📜 dateFormat.ts
|  |  |  |  |- 📜 firebase.ts
|  |  |  |- 📂 redux:
|  |  |  |  |- 📜 train.ts
|  |  |  |  |- 📜 userLogin.ts
|  |  |  |  |- 📜 store.ts
|  |  |  |  |- 📜 hooks.ts
|  |  |  |  |- 📜 backendApi.ts
|  |  |- 📜 next-env.d.ts
|  |  |- 📜 iris.csv : Sample CSV data
|  |  |- 📜 constants.ts
|  |  |- 📜 GlobalStyle.ts
|  |- 📜 next-env.d.ts
|  |- 📜 package.json
|  |- 📜 next.config.js
|  |- 📜 .eslintignore
|  |- 📜 .eslintrc.json
|  |- 📜 tsconfig.json
|  |- 📜 pnpm-lock.yaml
|  |- 📜 jest.config.js
```

