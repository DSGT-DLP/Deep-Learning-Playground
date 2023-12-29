# Architecture

## Training Architecture

```
📦 training
|  |- 📂 training:
|  |  |- 📂 core:
|  |  |  |- 📜 dl_model.py : torch model based on user specifications from drag and drop
|  |  |  |- 📜 authenticator.py
|  |  |  |- 📜 __init__.py
|  |  |  |- 📜 dataset.py : read in the dataset through URL or file upload
|  |  |  |- 📜 optimizer.py : what optimizer to use (ie: SGD or Adam for now)
|  |  |  |- 📜 trainer.py
|  |  |  |- 📜 criterion.py
|  |  |- 📂 routes:
|  |  |  |- 📂 tabular:
|  |  |  |  |- 📜 tabular.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |  |- 📜 schemas.py
|  |  |  |- 📂 datasets:
|  |  |  |  |- 📂 default:
|  |  |  |  |  |- 📜 __init__.py
|  |  |  |  |  |- 📜 columns.py
|  |  |  |  |  |- 📜 schemas.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |- 📂 image:
|  |  |  |  |- 📜 __init__.py
|  |  |  |  |- 📜 schemas.py
|  |  |  |  |- 📜 image.py
|  |  |  |- 📜 __init__.py
|  |  |  |- 📜 schemas.py
|  |  |- 📜 asgi.py
|  |  |- 📜 __init__.py
|  |  |- 📜 settings.py
|  |  |- 📜 urls.py
|  |  |- 📜 wsgi.py
|  |- 📜 pytest.ini
|  |- 📜 environment.yml
|  |- 📜 Dockerfile
|  |- 📜 pyproject.toml
|  |- 📜 README.md
|  |- 📜 cli.py
|  |- 📜 docker-compose.yml
|  |- 📜 docker-compose.prod.yml
|  |- 📜 manage.py
|  |- 📜 poetry.lock
```

## Frontend Architecture

```
📦 frontend
|  |- 📂 public:
|  |  |- 📂 images:
|  |  |  |- 📂 logos:
|  |  |  |  |- 📂 dlp_branding:
|  |  |  |  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |- 📜 flask-logo.png
|  |  |  |  |- 📜 github.png
|  |  |  |  |- 📜 python-logo.png
|  |  |  |  |- 📜 dsgt-logo-light.png
|  |  |  |  |- 📜 dsgt-logo-dark.png
|  |  |  |  |- 📜 pandas-logo.png
|  |  |  |  |- 📜 react-logo.png
|  |  |  |  |- 📜 pytorch-logo.png
|  |  |  |  |- 📜 dsgt-logo-white-back.png
|  |  |  |  |- 📜 google.png
|  |  |  |  |- 📜 aws-logo.png
|  |  |  |- 📂 wiki_images:
|  |  |  |  |- 📜 tanh_equation.png
|  |  |  |  |- 📜 avgpool_maxpool.gif
|  |  |  |  |- 📜 maxpool2d.gif
|  |  |  |  |- 📜 conv2d.gif
|  |  |  |  |- 📜 dropout_diagram.png
|  |  |  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |  |  |- 📜 conv2d2.gif
|  |  |  |  |- 📜 tanh_plot.png
|  |  |  |  |- 📜 batchnorm_diagram.png
|  |  |  |  |- 📜 sigmoid_equation.png
|  |  |  |- 📂 learn_mod_images:
|  |  |  |  |- 📜 lossExampleTable.png
|  |  |  |  |- 📜 neuronWithEquation.png
|  |  |  |  |- 📜 sigmoidfunction.png
|  |  |  |  |- 📜 ReLUactivation.png
|  |  |  |  |- 📜 neuralnet.png
|  |  |  |  |- 📜 sigmoidactivation.png
|  |  |  |  |- 📜 lossExample.png
|  |  |  |  |- 📜 tanhactivation.png
|  |  |  |  |- 📜 LeakyReLUactivation.png
|  |  |  |  |- 📜 neuron.png
|  |  |  |  |- 📜 binarystepactivation.png
|  |  |  |  |- 📜 robotImage.jpg
|  |  |  |  |- 📜 lossExampleEquation.png
|  |  |  |- 📜 demo_video.gif : GIF tutorial of a simple classification training session
|  |  |- 📜 index.html : Base HTML file that will be initially rendered
|  |  |- 📜 dlp-logo.ico : DLP Logo
|  |  |- 📜 manifest.json : Default React file for choosing icon based on
|  |  |- 📜 robots.txt
|  |- 📂 layer_docs:
|  |  |- 📜 ReLU.md : Doc for ReLU later
|  |  |- 📜 Softmax.md : Doc for Softmax layer
|  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |- 📜 Linear.md : Doc for Linear layer
|  |- 📂 src:
|  |  |- 📂 backend_outputs:
|  |  |  |- 📜 model.pt : Last model.pt output
|  |  |  |- 📜 model.pkl
|  |  |  |- 📜 my_deep_learning_model.onnx : Last ONNX file output
|  |  |- 📂 features:
|  |  |  |- 📂 LearnMod:
|  |  |  |  |- 📜 MCQuestion.tsx
|  |  |  |  |- 📜 LearningModulesContent.tsx
|  |  |  |  |- 📜 FRQuestion.tsx
|  |  |  |  |- 📜 ImageComponent.tsx
|  |  |  |  |- 📜 Exercise.tsx
|  |  |  |  |- 📜 ClassCard.tsx
|  |  |  |  |- 📜 ModulesSideBar.tsx
|  |  |  |- 📂 Feedback:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 feedbackApi.ts
|  |  |  |- 📂 Dashboard:
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TrainDataGrid.tsx
|  |  |  |  |  |- 📜 TrainBarChart.tsx
|  |  |  |  |  |- 📜 TrainDoughnutChart.tsx
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 dashboardApi.ts
|  |  |  |- 📂 Train:
|  |  |  |  |- 📂 types:
|  |  |  |  |  |- 📜 trainTypes.ts
|  |  |  |  |- 📂 constants:
|  |  |  |  |  |- 📜 trainConstants.ts
|  |  |  |  |- 📂 features:
|  |  |  |  |  |- 📂 Tabular:
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 tabularTypes.ts
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 tabularConstants.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 TabularFlow.tsx
|  |  |  |  |  |  |  |- 📜 TabularTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 TabularDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularReviewStep.tsx
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 tabularApi.ts
|  |  |  |  |  |  |  |- 📜 tabularActions.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |  |- 📂 Image:
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 imageTypes.ts
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 imageConstants.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 ImageReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 ImageFlow.tsx
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 imageActions.ts
|  |  |  |  |  |  |  |- 📜 imageApi.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 CreateTrainspace.tsx
|  |  |  |  |  |- 📜 DatasetStepLayout.tsx
|  |  |  |  |  |- 📜 TrainspaceLayout.tsx
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 trainspaceApi.ts
|  |  |  |  |  |- 📜 trainspaceSlice.ts
|  |  |  |- 📂 OpenAi:
|  |  |  |  |- 📜 openAiUtils.ts
|  |  |- 📂 common:
|  |  |  |- 📂 utils:
|  |  |  |  |- 📜 dateFormat.ts
|  |  |  |  |- 📜 dndHelpers.ts
|  |  |  |  |- 📜 firebase.ts
|  |  |  |- 📂 styles:
|  |  |  |  |- 📜 globals.css
|  |  |  |  |- 📜 Home.module.css
|  |  |  |- 📂 components:
|  |  |  |  |- 📜 ClientOnlyPortal.tsx
|  |  |  |  |- 📜 Spacer.tsx
|  |  |  |  |- 📜 EmailInput.tsx
|  |  |  |  |- 📜 NavBarMain.tsx
|  |  |  |  |- 📜 Footer.tsx
|  |  |  |  |- 📜 DlpTooltip.tsx
|  |  |  |  |- 📜 TitleText.tsx
|  |  |  |  |- 📜 HtmlTooltip.tsx
|  |  |  |- 📂 redux:
|  |  |  |  |- 📜 train.ts
|  |  |  |  |- 📜 backendApi.ts
|  |  |  |  |- 📜 userLogin.ts
|  |  |  |  |- 📜 hooks.ts
|  |  |  |  |- 📜 store.ts
|  |  |- 📂 pages:
|  |  |  |- 📂 train:
|  |  |  |  |- 📜 index.tsx
|  |  |  |  |- 📜 [train_space_id].tsx
|  |  |  |- 📜 settings.tsx
|  |  |  |- 📜 wiki.tsx
|  |  |  |- 📜 login.tsx
|  |  |  |- 📜 dashboard.tsx
|  |  |  |- 📜 _document.tsx
|  |  |  |- 📜 LearnContent.tsx
|  |  |  |- 📜 learn.tsx
|  |  |  |- 📜 feedback.tsx
|  |  |  |- 📜 forgot.tsx
|  |  |  |- 📜 _app.tsx
|  |  |  |- 📜 about.tsx
|  |  |- 📜 next-env.d.ts
|  |  |- 📜 iris.csv : Sample CSV data
|  |  |- 📜 constants.ts
|  |  |- 📜 GlobalStyle.ts
|  |- 📜 pnpm-lock.yaml
|  |- 📜 jest.config.js
|  |- 📜 next.config.js
|  |- 📜 next-env.d.ts
|  |- 📜 package.json
|  |- 📜 tsconfig.json
|  |- 📜 .eslintignore
|  |- 📜 .eslintrc.json
```

