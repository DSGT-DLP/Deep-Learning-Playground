# Architecture

## Training Architecture

```
📦 training
|  |- 📂 training:
|  |  |- 📂 core:
|  |  |  |- 📂 celery:
|  |  |  |  |- 📜 dataset.py : read in the dataset through URL or file upload
|  |  |  |  |- 📜 worker.py
|  |  |  |  |- 📜 dl_model.py : torch model based on user specifications from drag and drop
|  |  |  |  |- 📜 criterion.py
|  |  |  |  |- 📜 optimizer.py : what optimizer to use (ie: SGD or Adam for now)
|  |  |  |  |- 📜 trainer.py
|  |  |  |  |- 📜 Dockerfile
|  |  |  |  |- 📜 __init__.py
|  |  |  |  |- 📜 train_types.py
|  |  |  |- 📜 authenticator.py
|  |  |  |- 📜 __init__.py
|  |  |- 📂 routes:
|  |  |  |- 📂 datasets:
|  |  |  |  |- 📂 default:
|  |  |  |  |  |- 📜 schemas.py
|  |  |  |  |  |- 📜 columns.py
|  |  |  |  |  |- 📜 __init__.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |- 📂 tabular:
|  |  |  |  |- 📜 schemas.py
|  |  |  |  |- 📜 tabular.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |- 📂 audio:
|  |  |  |  |- 📜 audio.py
|  |  |  |  |- 📜 schemas.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |- 📂 image:
|  |  |  |  |- 📜 image.py
|  |  |  |  |- 📜 schemas.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |- 📜 schemas.py
|  |  |  |- 📜 __init__.py
|  |  |- 📂 middleware:
|  |  |  |- 📜 health_check_middleware.py
|  |  |  |- 📜 __init__.py
|  |  |- 📜 wsgi.py
|  |  |- 📜 asgi.py
|  |  |- 📜 settings.py
|  |  |- 📜 urls.py
|  |  |- 📜 constants.py : list of helpful constants
|  |  |- 📜 celery_app.py
|  |  |- 📜 __init__.py
|  |  |- 📜 celeryconfig.py
|  |- 📜 poetry.lock
|  |- 📜 environment.yml
|  |- 📜 README.md
|  |- 📜 pytest.ini
|  |- 📜 docker-compose.yml
|  |- 📜 manage.py
|  |- 📜 pyproject.toml
|  |- 📜 Dockerfile
|  |- 📜 cli.py
|  |- 📜 docker-compose.prod.yml
```

## Frontend Architecture

```
📦 frontend
|  |- 📂 layer_docs:
|  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |- 📜 Linear.md : Doc for Linear layer
|  |  |- 📜 Softmax.md : Doc for Softmax layer
|  |  |- 📜 ReLU.md : Doc for ReLU later
|  |- 📂 public:
|  |  |- 📂 images:
|  |  |  |- 📂 learn_mod_images:
|  |  |  |  |- 📜 lossExampleEquation.png
|  |  |  |  |- 📜 neuralnet.png
|  |  |  |  |- 📜 neuron.png
|  |  |  |  |- 📜 LeakyReLUactivation.png
|  |  |  |  |- 📜 sigmoidfunction.png
|  |  |  |  |- 📜 binarystepactivation.png
|  |  |  |  |- 📜 tanhactivation.png
|  |  |  |  |- 📜 lossExample.png
|  |  |  |  |- 📜 neuronWithEquation.png
|  |  |  |  |- 📜 sigmoidactivation.png
|  |  |  |  |- 📜 ReLUactivation.png
|  |  |  |  |- 📜 robotImage.jpg
|  |  |  |  |- 📜 lossExampleTable.png
|  |  |  |- 📂 wiki_images:
|  |  |  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |  |  |- 📜 sigmoid_equation.png
|  |  |  |  |- 📜 batchnorm_diagram.png
|  |  |  |  |- 📜 conv2d2.gif
|  |  |  |  |- 📜 avgpool_maxpool.gif
|  |  |  |  |- 📜 tanh_plot.png
|  |  |  |  |- 📜 dropout_diagram.png
|  |  |  |  |- 📜 tanh_equation.png
|  |  |  |  |- 📜 conv2d.gif
|  |  |  |  |- 📜 maxpool2d.gif
|  |  |  |- 📂 logos:
|  |  |  |  |- 📂 dlp_branding:
|  |  |  |  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |- 📜 pytorch-logo.png
|  |  |  |  |- 📜 dsgt-logo-light.png
|  |  |  |  |- 📜 dsgt-logo-white-back.png
|  |  |  |  |- 📜 flask-logo.png
|  |  |  |  |- 📜 github.png
|  |  |  |  |- 📜 dsgt-logo-dark.png
|  |  |  |  |- 📜 pandas-logo.png
|  |  |  |  |- 📜 aws-logo.png
|  |  |  |  |- 📜 react-logo.png
|  |  |  |  |- 📜 google.png
|  |  |  |  |- 📜 python-logo.png
|  |  |  |- 📜 demo_video.gif : GIF tutorial of a simple classification training session
|  |  |- 📜 dlp-logo.ico : DLP Logo
|  |  |- 📜 robots.txt
|  |  |- 📜 manifest.json : Default React file for choosing icon based on
|  |  |- 📜 index.html : Base HTML file that will be initially rendered
|  |- 📂 src:
|  |  |- 📂 backend_outputs:
|  |  |  |- 📜 model.pkl
|  |  |  |- 📜 my_deep_learning_model.onnx : Last ONNX file output
|  |  |  |- 📜 model.pt : Last model.pt output
|  |  |- 📂 common:
|  |  |  |- 📂 components:
|  |  |  |  |- 📜 ClientOnlyPortal.tsx
|  |  |  |  |- 📜 Footer.tsx
|  |  |  |  |- 📜 DlpTooltip.tsx
|  |  |  |  |- 📜 TitleText.tsx
|  |  |  |  |- 📜 NavBarMain.tsx
|  |  |  |  |- 📜 EmailInput.tsx
|  |  |  |  |- 📜 HtmlTooltip.tsx
|  |  |  |  |- 📜 Spacer.tsx
|  |  |  |- 📂 redux:
|  |  |  |  |- 📜 userLogin.ts
|  |  |  |  |- 📜 train.ts
|  |  |  |  |- 📜 store.ts
|  |  |  |  |- 📜 hooks.ts
|  |  |  |  |- 📜 backendApi.ts
|  |  |  |- 📂 styles:
|  |  |  |  |- 📜 Home.module.css
|  |  |  |  |- 📜 globals.css
|  |  |  |- 📂 utils:
|  |  |  |  |- 📜 dateFormat.ts
|  |  |  |  |- 📜 dndHelpers.ts
|  |  |  |  |- 📜 firebase.ts
|  |  |- 📂 __tests__:
|  |  |  |- 📂 common:
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TitleText.test.tsx
|  |  |- 📂 pages:
|  |  |  |- 📂 train:
|  |  |  |  |- 📜 index.tsx
|  |  |  |  |- 📜 [train_space_id].tsx
|  |  |  |  |- 📜 metrics_to_charts.tsx
|  |  |  |- 📜 wiki.tsx
|  |  |  |- 📜 _app.tsx
|  |  |  |- 📜 forgot.tsx
|  |  |  |- 📜 learn.tsx
|  |  |  |- 📜 settings.tsx
|  |  |  |- 📜 feedback.tsx
|  |  |  |- 📜 _document.tsx
|  |  |  |- 📜 dashboard.tsx
|  |  |  |- 📜 about.tsx
|  |  |  |- 📜 login.tsx
|  |  |  |- 📜 LearnContent.tsx
|  |  |- 📂 features:
|  |  |  |- 📂 Feedback:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 feedbackApi.ts
|  |  |  |- 📂 LearnMod:
|  |  |  |  |- 📜 MCQuestion.tsx
|  |  |  |  |- 📜 LearningModulesContent.tsx
|  |  |  |  |- 📜 ClassCard.tsx
|  |  |  |  |- 📜 Exercise.tsx
|  |  |  |  |- 📜 ModulesSideBar.tsx
|  |  |  |  |- 📜 ImageComponent.tsx
|  |  |  |  |- 📜 FRQuestion.tsx
|  |  |  |- 📂 Dashboard:
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TrainBarChart.tsx
|  |  |  |  |  |- 📜 TrainDataGrid.tsx
|  |  |  |  |  |- 📜 TrainDoughnutChart.tsx
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 dashboardApi.ts
|  |  |  |- 📂 Train:
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 DatasetStepLayout.tsx
|  |  |  |  |  |- 📜 TrainspaceLayout.tsx
|  |  |  |  |  |- 📜 CreateTrainspace.tsx
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 trainspaceSlice.ts
|  |  |  |  |  |- 📜 trainspaceApi.ts
|  |  |  |  |- 📂 types:
|  |  |  |  |  |- 📜 trainTypes.ts
|  |  |  |  |- 📂 constants:
|  |  |  |  |  |- 📜 trainConstants.ts
|  |  |  |  |- 📂 features:
|  |  |  |  |  |- 📂 Image:
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 ImageFlow.tsx
|  |  |  |  |  |  |  |- 📜 ImageTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 ImageParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageDatasetStep.tsx
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 imageApi.ts
|  |  |  |  |  |  |  |- 📜 imageActions.ts
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 imageTypes.ts
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 imageConstants.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |  |- 📂 Tabular:
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 TabularDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 TabularReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularFlow.tsx
|  |  |  |  |  |  |  |- 📜 TabularParametersStep.tsx
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 tabularActions.ts
|  |  |  |  |  |  |  |- 📜 tabularApi.ts
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 tabularTypes.ts
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 tabularConstants.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |- 📂 OpenAi:
|  |  |  |  |- 📜 openAiUtils.ts
|  |  |- 📜 iris.csv : Sample CSV data
|  |  |- 📜 next-env.d.ts
|  |  |- 📜 GlobalStyle.ts
|  |  |- 📜 constants.ts
|  |- 📜 tsconfig.json
|  |- 📜 .eslintrc.json
|  |- 📜 next.config.js
|  |- 📜 jest.config.ts
|  |- 📜 next-env.d.ts
|  |- 📜 .eslintignore
|  |- 📜 package.json
|  |- 📜 pnpm-lock.yaml
```

