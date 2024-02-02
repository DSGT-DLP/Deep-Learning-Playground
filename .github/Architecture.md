# Architecture

## Training Architecture

```
📦 training
|  |- 📂 training:
|  |  |- 📂 core:
|  |  |  |- 📜 optimizer.py : what optimizer to use (ie: SGD or Adam for now)
|  |  |  |- 📜 dl_model.py : torch model based on user specifications from drag and drop
|  |  |  |- 📜 dataset.py : read in the dataset through URL or file upload
|  |  |  |- 📜 authenticator.py
|  |  |  |- 📜 trainer.py
|  |  |  |- 📜 criterion.py
|  |  |  |- 📜 __init__.py
|  |  |- 📂 routes:
|  |  |  |- 📂 image:
|  |  |  |  |- 📜 schemas.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |  |- 📜 image.py
|  |  |  |- 📂 tabular:
|  |  |  |  |- 📜 tabular.py
|  |  |  |  |- 📜 schemas.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |- 📂 datasets:
|  |  |  |  |- 📂 default:
|  |  |  |  |  |- 📜 schemas.py
|  |  |  |  |  |- 📜 columns.py
|  |  |  |  |  |- 📜 __init__.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |- 📜 schemas.py
|  |  |  |- 📜 __init__.py
|  |  |- 📜 settings.py
|  |  |- 📜 asgi.py
|  |  |- 📜 wsgi.py
|  |  |- 📜 urls.py
|  |  |- 📜 __init__.py
|  |- 📜 cli.py
|  |- 📜 Dockerfile
|  |- 📜 README.md
|  |- 📜 docker-compose.yml
|  |- 📜 manage.py
|  |- 📜 poetry.lock
|  |- 📜 Dockerfile.prod
|  |- 📜 environment.yml
|  |- 📜 pytest.ini
|  |- 📜 pyproject.toml
|  |- 📜 docker-compose.prod.yml
```

## Frontend Architecture

```
📦 frontend
|  |- 📂 src:
|  |  |- 📂 pages:
|  |  |  |- 📂 train:
|  |  |  |  |- 📜 [train_space_id].tsx
|  |  |  |  |- 📜 index.tsx
|  |  |  |- 📜 wiki.tsx
|  |  |  |- 📜 _app.tsx
|  |  |  |- 📜 learn.tsx
|  |  |  |- 📜 LearnContent.tsx
|  |  |  |- 📜 forgot.tsx
|  |  |  |- 📜 feedback.tsx
|  |  |  |- 📜 about.tsx
|  |  |  |- 📜 dashboard.tsx
|  |  |  |- 📜 login.tsx
|  |  |  |- 📜 _document.tsx
|  |  |  |- 📜 settings.tsx
|  |  |- 📂 common:
|  |  |  |- 📂 utils:
|  |  |  |  |- 📜 dateFormat.ts
|  |  |  |  |- 📜 firebase.ts
|  |  |  |  |- 📜 dndHelpers.ts
|  |  |  |- 📂 redux:
|  |  |  |  |- 📜 hooks.ts
|  |  |  |  |- 📜 store.ts
|  |  |  |  |- 📜 userLogin.ts
|  |  |  |  |- 📜 train.ts
|  |  |  |  |- 📜 backendApi.ts
|  |  |  |- 📂 styles:
|  |  |  |  |- 📜 globals.css
|  |  |  |  |- 📜 Home.module.css
|  |  |  |- 📂 components:
|  |  |  |  |- 📜 HtmlTooltip.tsx
|  |  |  |  |- 📜 DlpTooltip.tsx
|  |  |  |  |- 📜 TitleText.tsx
|  |  |  |  |- 📜 ClientOnlyPortal.tsx
|  |  |  |  |- 📜 EmailInput.tsx
|  |  |  |  |- 📜 Spacer.tsx
|  |  |  |  |- 📜 Footer.tsx
|  |  |  |  |- 📜 NavBarMain.tsx
|  |  |- 📂 features:
|  |  |  |- 📂 Train:
|  |  |  |  |- 📂 features:
|  |  |  |  |  |- 📂 Tabular:
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 tabularActions.ts
|  |  |  |  |  |  |  |- 📜 tabularApi.ts
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 tabularConstants.ts
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 tabularTypes.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 TabularDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 TabularFlow.tsx
|  |  |  |  |  |  |  |- 📜 TabularParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularReviewStep.tsx
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |  |- 📂 Image:
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 imageActions.ts
|  |  |  |  |  |  |  |- 📜 imageApi.ts
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 imageConstants.ts
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 imageTypes.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 ImageReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 ImageFlow.tsx
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 trainspaceApi.ts
|  |  |  |  |  |- 📜 trainspaceSlice.ts
|  |  |  |  |- 📂 constants:
|  |  |  |  |  |- 📜 trainConstants.ts
|  |  |  |  |- 📂 types:
|  |  |  |  |  |- 📜 trainTypes.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 CreateTrainspace.tsx
|  |  |  |  |  |- 📜 TrainspaceLayout.tsx
|  |  |  |  |  |- 📜 DatasetStepLayout.tsx
|  |  |  |- 📂 Feedback:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 feedbackApi.ts
|  |  |  |- 📂 OpenAi:
|  |  |  |  |- 📜 openAiUtils.ts
|  |  |  |- 📂 LearnMod:
|  |  |  |  |- 📜 MCQuestion.tsx
|  |  |  |  |- 📜 ModulesSideBar.tsx
|  |  |  |  |- 📜 ImageComponent.tsx
|  |  |  |  |- 📜 LearningModulesContent.tsx
|  |  |  |  |- 📜 FRQuestion.tsx
|  |  |  |  |- 📜 Exercise.tsx
|  |  |  |  |- 📜 ClassCard.tsx
|  |  |  |- 📂 Dashboard:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 dashboardApi.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TrainDoughnutChart.tsx
|  |  |  |  |  |- 📜 TrainBarChart.tsx
|  |  |  |  |  |- 📜 TrainDataGrid.tsx
|  |  |- 📂 backend_outputs:
|  |  |  |- 📜 my_deep_learning_model.onnx : Last ONNX file output
|  |  |  |- 📜 model.pt : Last model.pt output
|  |  |  |- 📜 model.pkl
|  |  |- 📜 GlobalStyle.ts
|  |  |- 📜 iris.csv : Sample CSV data
|  |  |- 📜 constants.ts
|  |  |- 📜 next-env.d.ts
|  |- 📂 public:
|  |  |- 📂 images:
|  |  |  |- 📂 wiki_images:
|  |  |  |  |- 📜 maxpool2d.gif
|  |  |  |  |- 📜 tanh_equation.png
|  |  |  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |  |  |- 📜 conv2d2.gif
|  |  |  |  |- 📜 dropout_diagram.png
|  |  |  |  |- 📜 tanh_plot.png
|  |  |  |  |- 📜 avgpool_maxpool.gif
|  |  |  |  |- 📜 batchnorm_diagram.png
|  |  |  |  |- 📜 conv2d.gif
|  |  |  |  |- 📜 sigmoid_equation.png
|  |  |  |- 📂 logos:
|  |  |  |  |- 📂 dlp_branding:
|  |  |  |  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |- 📜 aws-logo.png
|  |  |  |  |- 📜 pytorch-logo.png
|  |  |  |  |- 📜 python-logo.png
|  |  |  |  |- 📜 react-logo.png
|  |  |  |  |- 📜 github.png
|  |  |  |  |- 📜 pandas-logo.png
|  |  |  |  |- 📜 google.png
|  |  |  |  |- 📜 dsgt-logo-white-back.png
|  |  |  |  |- 📜 dsgt-logo-dark.png
|  |  |  |  |- 📜 flask-logo.png
|  |  |  |  |- 📜 dsgt-logo-light.png
|  |  |  |- 📂 learn_mod_images:
|  |  |  |  |- 📜 lossExample.png
|  |  |  |  |- 📜 neuronWithEquation.png
|  |  |  |  |- 📜 neuralnet.png
|  |  |  |  |- 📜 robotImage.jpg
|  |  |  |  |- 📜 sigmoidfunction.png
|  |  |  |  |- 📜 lossExampleTable.png
|  |  |  |  |- 📜 LeakyReLUactivation.png
|  |  |  |  |- 📜 lossExampleEquation.png
|  |  |  |  |- 📜 ReLUactivation.png
|  |  |  |  |- 📜 neuron.png
|  |  |  |  |- 📜 tanhactivation.png
|  |  |  |  |- 📜 sigmoidactivation.png
|  |  |  |  |- 📜 binarystepactivation.png
|  |  |  |- 📜 demo_video.gif : GIF tutorial of a simple classification training session
|  |  |- 📜 manifest.json : Default React file for choosing icon based on
|  |  |- 📜 index.html : Base HTML file that will be initially rendered
|  |  |- 📜 robots.txt
|  |  |- 📜 dlp-logo.ico : DLP Logo
|  |- 📂 layer_docs:
|  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |- 📜 Softmax.md : Doc for Softmax layer
|  |  |- 📜 Linear.md : Doc for Linear layer
|  |  |- 📜 ReLU.md : Doc for ReLU later
|  |- 📜 .eslintrc.json
|  |- 📜 .eslintignore
|  |- 📜 pnpm-lock.yaml
|  |- 📜 next.config.js
|  |- 📜 jest.config.js
|  |- 📜 package.json
|  |- 📜 tsconfig.json
|  |- 📜 next-env.d.ts
```

