# Architecture

## Training Architecture

```
📦 training
|  |- 📂 training:
|  |  |- 📂 core:
|  |  |  |- 📜 dataset.py : read in the dataset through URL or file upload
|  |  |  |- 📜 criterion.py
|  |  |  |- 📜 optimizer.py : what optimizer to use (ie: SGD or Adam for now)
|  |  |  |- 📜 __init__.py
|  |  |  |- 📜 dl_model.py : torch model based on user specifications from drag and drop
|  |  |  |- 📜 trainer.py
|  |  |  |- 📜 authenticator.py
|  |  |- 📂 routes:
|  |  |  |- 📂 datasets:
|  |  |  |  |- 📂 default:
|  |  |  |  |  |- 📜 schemas.py
|  |  |  |  |  |- 📜 __init__.py
|  |  |  |  |  |- 📜 columns.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |- 📂 image:
|  |  |  |  |- 📜 image.py
|  |  |  |  |- 📜 schemas.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |- 📂 tabular:
|  |  |  |  |- 📜 tabular.py
|  |  |  |  |- 📜 schemas.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |- 📜 schemas.py
|  |  |  |- 📜 __init__.py
|  |  |- 📂 middleware:
|  |  |  |- 📜 health_check_middleware.py
|  |  |  |- 📜 __init__.py
|  |  |- 📜 urls.py
|  |  |- 📜 asgi.py
|  |  |- 📜 __init__.py
|  |  |- 📜 wsgi.py
|  |  |- 📜 settings.py
|  |- 📜 README.md
|  |- 📜 docker-compose.prod.yml
|  |- 📜 pytest.ini
|  |- 📜 poetry.lock
|  |- 📜 cli.py
|  |- 📜 pyproject.toml
|  |- 📜 environment.yml
|  |- 📜 Dockerfile
|  |- 📜 Dockerfile.prod
|  |- 📜 manage.py
|  |- 📜 docker-compose.yml
```

## Frontend Architecture

```
📦 frontend
|  |- 📂 src:
|  |  |- 📂 pages:
|  |  |  |- 📂 train:
|  |  |  |  |- 📜 [train_space_id].tsx
|  |  |  |  |- 📜 index.tsx
|  |  |  |- 📜 settings.tsx
|  |  |  |- 📜 feedback.tsx
|  |  |  |- 📜 forgot.tsx
|  |  |  |- 📜 learn.tsx
|  |  |  |- 📜 dashboard.tsx
|  |  |  |- 📜 login.tsx
|  |  |  |- 📜 _document.tsx
|  |  |  |- 📜 _app.tsx
|  |  |  |- 📜 about.tsx
|  |  |  |- 📜 wiki.tsx
|  |  |  |- 📜 LearnContent.tsx
|  |  |- 📂 features:
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
|  |  |  |  |- 📂 constants:
|  |  |  |  |  |- 📜 trainConstants.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 DatasetStepLayout.tsx
|  |  |  |  |  |- 📜 CreateTrainspace.tsx
|  |  |  |  |  |- 📜 TrainspaceLayout.tsx
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 trainspaceApi.ts
|  |  |  |  |  |- 📜 trainspaceSlice.ts
|  |  |  |  |- 📂 features:
|  |  |  |  |  |- 📂 Image:
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 imageConstants.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 ImageTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 ImageFlow.tsx
|  |  |  |  |  |  |  |- 📜 ImageReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageDatasetStep.tsx
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 imageApi.ts
|  |  |  |  |  |  |  |- 📜 imageActions.ts
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 imageTypes.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |  |- 📂 Tabular:
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 tabularConstants.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 TabularTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 TabularParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularFlow.tsx
|  |  |  |  |  |  |  |- 📜 TabularReviewStep.tsx
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 tabularActions.ts
|  |  |  |  |  |  |  |- 📜 tabularApi.ts
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 tabularTypes.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |- 📂 types:
|  |  |  |  |  |- 📜 trainTypes.ts
|  |  |  |- 📂 OpenAi:
|  |  |  |  |- 📜 openAiUtils.ts
|  |  |  |- 📂 LearnMod:
|  |  |  |  |- 📜 ModulesSideBar.tsx
|  |  |  |  |- 📜 Exercise.tsx
|  |  |  |  |- 📜 LearningModulesContent.tsx
|  |  |  |  |- 📜 FRQuestion.tsx
|  |  |  |  |- 📜 ImageComponent.tsx
|  |  |  |  |- 📜 ClassCard.tsx
|  |  |  |  |- 📜 MCQuestion.tsx
|  |  |- 📂 backend_outputs:
|  |  |  |- 📜 model.pkl
|  |  |  |- 📜 my_deep_learning_model.onnx : Last ONNX file output
|  |  |  |- 📜 model.pt : Last model.pt output
|  |  |- 📂 common:
|  |  |  |- 📂 components:
|  |  |  |  |- 📜 Spacer.tsx
|  |  |  |  |- 📜 Footer.tsx
|  |  |  |  |- 📜 DlpTooltip.tsx
|  |  |  |  |- 📜 EmailInput.tsx
|  |  |  |  |- 📜 HtmlTooltip.tsx
|  |  |  |  |- 📜 NavBarMain.tsx
|  |  |  |  |- 📜 TitleText.tsx
|  |  |  |  |- 📜 ClientOnlyPortal.tsx
|  |  |  |- 📂 redux:
|  |  |  |  |- 📜 store.ts
|  |  |  |  |- 📜 train.ts
|  |  |  |  |- 📜 userLogin.ts
|  |  |  |  |- 📜 hooks.ts
|  |  |  |  |- 📜 backendApi.ts
|  |  |  |- 📂 utils:
|  |  |  |  |- 📜 dateFormat.ts
|  |  |  |  |- 📜 firebase.ts
|  |  |  |  |- 📜 dndHelpers.ts
|  |  |  |- 📂 styles:
|  |  |  |  |- 📜 Home.module.css
|  |  |  |  |- 📜 globals.css
|  |  |- 📜 next-env.d.ts
|  |  |- 📜 GlobalStyle.ts
|  |  |- 📜 iris.csv : Sample CSV data
|  |  |- 📜 constants.ts
|  |- 📂 layer_docs:
|  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |- 📜 Linear.md : Doc for Linear layer
|  |  |- 📜 Softmax.md : Doc for Softmax layer
|  |  |- 📜 ReLU.md : Doc for ReLU later
|  |- 📂 public:
|  |  |- 📂 images:
|  |  |  |- 📂 wiki_images:
|  |  |  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |  |  |- 📜 tanh_plot.png
|  |  |  |  |- 📜 conv2d.gif
|  |  |  |  |- 📜 conv2d2.gif
|  |  |  |  |- 📜 avgpool_maxpool.gif
|  |  |  |  |- 📜 sigmoid_equation.png
|  |  |  |  |- 📜 batchnorm_diagram.png
|  |  |  |  |- 📜 maxpool2d.gif
|  |  |  |  |- 📜 tanh_equation.png
|  |  |  |  |- 📜 dropout_diagram.png
|  |  |  |- 📂 learn_mod_images:
|  |  |  |  |- 📜 neuronWithEquation.png
|  |  |  |  |- 📜 robotImage.jpg
|  |  |  |  |- 📜 neuralnet.png
|  |  |  |  |- 📜 sigmoidactivation.png
|  |  |  |  |- 📜 lossExample.png
|  |  |  |  |- 📜 binarystepactivation.png
|  |  |  |  |- 📜 tanhactivation.png
|  |  |  |  |- 📜 LeakyReLUactivation.png
|  |  |  |  |- 📜 sigmoidfunction.png
|  |  |  |  |- 📜 lossExampleTable.png
|  |  |  |  |- 📜 lossExampleEquation.png
|  |  |  |  |- 📜 neuron.png
|  |  |  |  |- 📜 ReLUactivation.png
|  |  |  |- 📂 logos:
|  |  |  |  |- 📂 dlp_branding:
|  |  |  |  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |- 📜 dsgt-logo-white-back.png
|  |  |  |  |- 📜 pytorch-logo.png
|  |  |  |  |- 📜 google.png
|  |  |  |  |- 📜 flask-logo.png
|  |  |  |  |- 📜 dsgt-logo-dark.png
|  |  |  |  |- 📜 pandas-logo.png
|  |  |  |  |- 📜 dsgt-logo-light.png
|  |  |  |  |- 📜 github.png
|  |  |  |  |- 📜 react-logo.png
|  |  |  |  |- 📜 python-logo.png
|  |  |  |  |- 📜 aws-logo.png
|  |  |  |- 📜 demo_video.gif : GIF tutorial of a simple classification training session
|  |  |- 📜 manifest.json : Default React file for choosing icon based on
|  |  |- 📜 dlp-logo.ico : DLP Logo
|  |  |- 📜 index.html : Base HTML file that will be initially rendered
|  |  |- 📜 robots.txt
|  |- 📜 .eslintrc.json
|  |- 📜 .eslintignore
|  |- 📜 pnpm-lock.yaml
|  |- 📜 package.json
|  |- 📜 next.config.js
|  |- 📜 jest.config.js
|  |- 📜 next-env.d.ts
|  |- 📜 tsconfig.json
```

