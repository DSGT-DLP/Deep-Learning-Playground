# Architecture

## Training Architecture

```
📦 training
|  |- 📂 training:
|  |  |- 📂 core:
|  |  |  |- 📜 dl_model.py : torch model based on user specifications from drag and drop
|  |  |  |- 📜 trainer.py
|  |  |  |- 📜 criterion.py
|  |  |  |- 📜 dataset.py : read in the dataset through URL or file upload
|  |  |  |- 📜 authenticator.py
|  |  |  |- 📜 __init__.py
|  |  |  |- 📜 optimizer.py : what optimizer to use (ie: SGD or Adam for now)
|  |  |- 📂 routes:
|  |  |  |- 📂 datasets:
|  |  |  |  |- 📂 default:
|  |  |  |  |  |- 📜 columns.py
|  |  |  |  |  |- 📜 schemas.py
|  |  |  |  |  |- 📜 __init__.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |- 📂 tabular:
|  |  |  |  |- 📜 tabular.py
|  |  |  |  |- 📜 schemas.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |- 📂 image:
|  |  |  |  |- 📜 image.py
|  |  |  |  |- 📜 schemas.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |- 📜 schemas.py
|  |  |  |- 📜 __init__.py
|  |  |- 📜 wsgi.py
|  |  |- 📜 settings.py
|  |  |- 📜 urls.py
|  |  |- 📜 __init__.py
|  |  |- 📜 asgi.py
|  |- 📜 manage.py
|  |- 📜 docker-compose.yml
|  |- 📜 poetry.lock
|  |- 📜 docker-compose.prod.yml
|  |- 📜 environment.yml
|  |- 📜 pytest.ini
|  |- 📜 Dockerfile
|  |- 📜 cli.py
|  |- 📜 pyproject.toml
|  |- 📜 README.md
|  |- 📜 Dockerfile.prod
```

## Frontend Architecture

```
📦 frontend
|  |- 📂 src:
|  |  |- 📂 features:
|  |  |  |- 📂 OpenAi:
|  |  |  |  |- 📜 openAiUtils.ts
|  |  |  |- 📂 Dashboard:
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TrainBarChart.tsx
|  |  |  |  |  |- 📜 TrainDoughnutChart.tsx
|  |  |  |  |  |- 📜 TrainDataGrid.tsx
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 dashboardApi.ts
|  |  |  |- 📂 Feedback:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 feedbackApi.ts
|  |  |  |- 📂 LearnMod:
|  |  |  |  |- 📜 ClassCard.tsx
|  |  |  |  |- 📜 Exercise.tsx
|  |  |  |  |- 📜 FRQuestion.tsx
|  |  |  |  |- 📜 ImageComponent.tsx
|  |  |  |  |- 📜 MCQuestion.tsx
|  |  |  |  |- 📜 ModulesSideBar.tsx
|  |  |  |  |- 📜 LearningModulesContent.tsx
|  |  |  |- 📂 Train:
|  |  |  |  |- 📂 features:
|  |  |  |  |  |- 📂 Image:
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 ImageFlow.tsx
|  |  |  |  |  |  |  |- 📜 ImageDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageTrainspace.tsx
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 imageActions.ts
|  |  |  |  |  |  |  |- 📜 imageApi.ts
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 imageConstants.ts
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 imageTypes.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |  |- 📂 Tabular:
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 TabularParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 TabularDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularFlow.tsx
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 tabularActions.ts
|  |  |  |  |  |  |  |- 📜 tabularApi.ts
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 tabularConstants.ts
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 tabularTypes.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 DatasetStepLayout.tsx
|  |  |  |  |  |- 📜 TrainspaceLayout.tsx
|  |  |  |  |  |- 📜 CreateTrainspace.tsx
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 trainspaceSlice.ts
|  |  |  |  |  |- 📜 trainspaceApi.ts
|  |  |  |  |- 📂 constants:
|  |  |  |  |  |- 📜 trainConstants.ts
|  |  |  |  |- 📂 types:
|  |  |  |  |  |- 📜 trainTypes.ts
|  |  |- 📂 pages:
|  |  |  |- 📂 train:
|  |  |  |  |- 📜 [train_space_id].tsx
|  |  |  |  |- 📜 index.tsx
|  |  |  |- 📜 about.tsx
|  |  |  |- 📜 learn.tsx
|  |  |  |- 📜 login.tsx
|  |  |  |- 📜 settings.tsx
|  |  |  |- 📜 _app.tsx
|  |  |  |- 📜 dashboard.tsx
|  |  |  |- 📜 LearnContent.tsx
|  |  |  |- 📜 forgot.tsx
|  |  |  |- 📜 wiki.tsx
|  |  |  |- 📜 feedback.tsx
|  |  |  |- 📜 _document.tsx
|  |  |- 📂 backend_outputs:
|  |  |  |- 📜 my_deep_learning_model.onnx : Last ONNX file output
|  |  |  |- 📜 model.pt : Last model.pt output
|  |  |  |- 📜 model.pkl
|  |  |- 📂 common:
|  |  |  |- 📂 components:
|  |  |  |  |- 📜 EmailInput.tsx
|  |  |  |  |- 📜 HtmlTooltip.tsx
|  |  |  |  |- 📜 NavBarMain.tsx
|  |  |  |  |- 📜 TitleText.tsx
|  |  |  |  |- 📜 DlpTooltip.tsx
|  |  |  |  |- 📜 ClientOnlyPortal.tsx
|  |  |  |  |- 📜 Footer.tsx
|  |  |  |  |- 📜 Spacer.tsx
|  |  |  |- 📂 redux:
|  |  |  |  |- 📜 userLogin.ts
|  |  |  |  |- 📜 store.ts
|  |  |  |  |- 📜 hooks.ts
|  |  |  |  |- 📜 train.ts
|  |  |  |  |- 📜 backendApi.ts
|  |  |  |- 📂 utils:
|  |  |  |  |- 📜 firebase.ts
|  |  |  |  |- 📜 dndHelpers.ts
|  |  |  |  |- 📜 dateFormat.ts
|  |  |  |- 📂 styles:
|  |  |  |  |- 📜 globals.css
|  |  |  |  |- 📜 Home.module.css
|  |  |- 📜 constants.ts
|  |  |- 📜 iris.csv : Sample CSV data
|  |  |- 📜 GlobalStyle.ts
|  |  |- 📜 next-env.d.ts
|  |- 📂 layer_docs:
|  |  |- 📜 ReLU.md : Doc for ReLU later
|  |  |- 📜 Softmax.md : Doc for Softmax layer
|  |  |- 📜 Linear.md : Doc for Linear layer
|  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |- 📂 public:
|  |  |- 📂 images:
|  |  |  |- 📂 wiki_images:
|  |  |  |  |- 📜 conv2d2.gif
|  |  |  |  |- 📜 batchnorm_diagram.png
|  |  |  |  |- 📜 conv2d.gif
|  |  |  |  |- 📜 avgpool_maxpool.gif
|  |  |  |  |- 📜 tanh_plot.png
|  |  |  |  |- 📜 maxpool2d.gif
|  |  |  |  |- 📜 sigmoid_equation.png
|  |  |  |  |- 📜 tanh_equation.png
|  |  |  |  |- 📜 dropout_diagram.png
|  |  |  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |  |- 📂 learn_mod_images:
|  |  |  |  |- 📜 LeakyReLUactivation.png
|  |  |  |  |- 📜 neuronWithEquation.png
|  |  |  |  |- 📜 lossExampleEquation.png
|  |  |  |  |- 📜 sigmoidactivation.png
|  |  |  |  |- 📜 lossExample.png
|  |  |  |  |- 📜 robotImage.jpg
|  |  |  |  |- 📜 sigmoidfunction.png
|  |  |  |  |- 📜 lossExampleTable.png
|  |  |  |  |- 📜 neuralnet.png
|  |  |  |  |- 📜 binarystepactivation.png
|  |  |  |  |- 📜 tanhactivation.png
|  |  |  |  |- 📜 neuron.png
|  |  |  |  |- 📜 ReLUactivation.png
|  |  |  |- 📂 logos:
|  |  |  |  |- 📂 dlp_branding:
|  |  |  |  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |- 📜 dsgt-logo-dark.png
|  |  |  |  |- 📜 react-logo.png
|  |  |  |  |- 📜 dsgt-logo-light.png
|  |  |  |  |- 📜 github.png
|  |  |  |  |- 📜 aws-logo.png
|  |  |  |  |- 📜 flask-logo.png
|  |  |  |  |- 📜 pandas-logo.png
|  |  |  |  |- 📜 pytorch-logo.png
|  |  |  |  |- 📜 python-logo.png
|  |  |  |  |- 📜 google.png
|  |  |  |  |- 📜 dsgt-logo-white-back.png
|  |  |  |- 📜 demo_video.gif : GIF tutorial of a simple classification training session
|  |  |- 📜 dlp-logo.ico : DLP Logo
|  |  |- 📜 robots.txt
|  |  |- 📜 index.html : Base HTML file that will be initially rendered
|  |  |- 📜 manifest.json : Default React file for choosing icon based on
|  |- 📜 package.json
|  |- 📜 .eslintrc.json
|  |- 📜 pnpm-lock.yaml
|  |- 📜 .eslintignore
|  |- 📜 next.config.js
|  |- 📜 jest.config.js
|  |- 📜 next-env.d.ts
|  |- 📜 tsconfig.json
```

