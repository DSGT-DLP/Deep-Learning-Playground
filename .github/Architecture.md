# Architecture

## Training Architecture

```
📦 training
|  |- 📂 training:
|  |  |- 📂 middleware:
|  |  |  |- 📜 __init__.py
|  |  |  |- 📜 health_check_middleware.py
|  |  |- 📂 routes:
|  |  |  |- 📂 datasets:
|  |  |  |  |- 📂 default:
|  |  |  |  |  |- 📜 schemas.py
|  |  |  |  |  |- 📜 __init__.py
|  |  |  |  |  |- 📜 columns.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |- 📂 image:
|  |  |  |  |- 📜 schemas.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |  |- 📜 image.py
|  |  |  |- 📂 audio:
|  |  |  |  |- 📜 schemas.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |  |- 📜 audio.py
|  |  |  |- 📂 tabular:
|  |  |  |  |- 📜 schemas.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |  |- 📜 tabular.py
|  |  |  |- 📜 schemas.py
|  |  |  |- 📜 __init__.py
|  |  |- 📂 core:
|  |  |  |- 📂 celery:
|  |  |  |  |- 📜 Dockerfile
|  |  |  |  |- 📜 __init__.py
|  |  |  |  |- 📜 optimizer.py : what optimizer to use (ie: SGD or Adam for now)
|  |  |  |  |- 📜 dataset.py : read in the dataset through URL or file upload
|  |  |  |  |- 📜 train_types.py
|  |  |  |  |- 📜 trainer.py
|  |  |  |  |- 📜 dl_model.py : torch model based on user specifications from drag and drop
|  |  |  |  |- 📜 criterion.py
|  |  |  |  |- 📜 worker.py
|  |  |  |- 📜 __init__.py
|  |  |  |- 📜 authenticator.py
|  |  |- 📜 __init__.py
|  |  |- 📜 wsgi.py
|  |  |- 📜 celeryconfig.py
|  |  |- 📜 settings.py
|  |  |- 📜 asgi.py
|  |  |- 📜 urls.py
|  |  |- 📜 celery_app.py
|  |  |- 📜 constants.py : list of helpful constants
|  |- 📜 pytest.ini
|  |- 📜 cli.py
|  |- 📜 Dockerfile
|  |- 📜 README.md
|  |- 📜 docker-compose.yml
|  |- 📜 docker-compose.prod.yml
|  |- 📜 pyproject.toml
|  |- 📜 environment.yml
|  |- 📜 poetry.lock
|  |- 📜 manage.py
```

## Frontend Architecture

```
📦 frontend
|  |- 📂 public:
|  |  |- 📂 images:
|  |  |  |- 📂 learn_mod_images:
|  |  |  |  |- 📜 neuralnet.png
|  |  |  |  |- 📜 ReLUactivation.png
|  |  |  |  |- 📜 binarystepactivation.png
|  |  |  |  |- 📜 neuron.png
|  |  |  |  |- 📜 sigmoidactivation.png
|  |  |  |  |- 📜 robotImage.jpg
|  |  |  |  |- 📜 LeakyReLUactivation.png
|  |  |  |  |- 📜 neuronWithEquation.png
|  |  |  |  |- 📜 lossExampleTable.png
|  |  |  |  |- 📜 lossExampleEquation.png
|  |  |  |  |- 📜 sigmoidfunction.png
|  |  |  |  |- 📜 tanhactivation.png
|  |  |  |  |- 📜 lossExample.png
|  |  |  |- 📂 logos:
|  |  |  |  |- 📂 dlp_branding:
|  |  |  |  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |- 📜 dsgt-logo-light.png
|  |  |  |  |- 📜 pytorch-logo.png
|  |  |  |  |- 📜 python-logo.png
|  |  |  |  |- 📜 google.png
|  |  |  |  |- 📜 aws-logo.png
|  |  |  |  |- 📜 github.png
|  |  |  |  |- 📜 flask-logo.png
|  |  |  |  |- 📜 dsgt-logo-white-back.png
|  |  |  |  |- 📜 dsgt-logo-dark.png
|  |  |  |  |- 📜 react-logo.png
|  |  |  |  |- 📜 pandas-logo.png
|  |  |  |- 📂 wiki_images:
|  |  |  |  |- 📜 sigmoid_equation.png
|  |  |  |  |- 📜 conv2d.gif
|  |  |  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |  |  |- 📜 batchnorm_diagram.png
|  |  |  |  |- 📜 dropout_diagram.png
|  |  |  |  |- 📜 maxpool2d.gif
|  |  |  |  |- 📜 conv2d2.gif
|  |  |  |  |- 📜 tanh_equation.png
|  |  |  |  |- 📜 avgpool_maxpool.gif
|  |  |  |  |- 📜 tanh_plot.png
|  |  |  |- 📜 demo_video.gif : GIF tutorial of a simple classification training session
|  |  |- 📜 manifest.json : Default React file for choosing icon based on
|  |  |- 📜 robots.txt
|  |  |- 📜 index.html : Base HTML file that will be initially rendered
|  |  |- 📜 dlp-logo.ico : DLP Logo
|  |- 📂 layer_docs:
|  |  |- 📜 Softmax.md : Doc for Softmax layer
|  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |- 📜 Linear.md : Doc for Linear layer
|  |  |- 📜 ReLU.md : Doc for ReLU later
|  |- 📂 src:
|  |  |- 📂 pages:
|  |  |  |- 📂 train:
|  |  |  |  |- 📜 [train_space_id].tsx
|  |  |  |  |- 📜 index.tsx
|  |  |  |  |- 📜 metrics_to_charts.tsx
|  |  |  |- 📜 settings.tsx
|  |  |  |- 📜 dashboard.tsx
|  |  |  |- 📜 login.tsx
|  |  |  |- 📜 learn.tsx
|  |  |  |- 📜 _app.tsx
|  |  |  |- 📜 _document.tsx
|  |  |  |- 📜 LearnContent.tsx
|  |  |  |- 📜 feedback.tsx
|  |  |  |- 📜 forgot.tsx
|  |  |  |- 📜 wiki.tsx
|  |  |  |- 📜 about.tsx
|  |  |- 📂 features:
|  |  |  |- 📂 Train:
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 DatasetStepLayout.tsx
|  |  |  |  |  |- 📜 CreateTrainspace.tsx
|  |  |  |  |  |- 📜 TrainspaceLayout.tsx
|  |  |  |  |- 📂 features:
|  |  |  |  |  |- 📂 Image:
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 ImageFlow.tsx
|  |  |  |  |  |  |  |- 📜 ImageParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageTrainspace.tsx
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 imageConstants.ts
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 imageTypes.ts
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 imageApi.ts
|  |  |  |  |  |  |  |- 📜 imageActions.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |  |- 📂 Tabular:
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 TabularParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularFlow.tsx
|  |  |  |  |  |  |  |- 📜 TabularTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 TabularReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularDatasetStep.tsx
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 tabularConstants.ts
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 tabularTypes.ts
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 tabularActions.ts
|  |  |  |  |  |  |  |- 📜 tabularApi.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |- 📂 constants:
|  |  |  |  |  |- 📜 trainConstants.ts
|  |  |  |  |- 📂 types:
|  |  |  |  |  |- 📜 trainTypes.ts
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 trainspaceApi.ts
|  |  |  |  |  |- 📜 trainspaceSlice.ts
|  |  |  |- 📂 LearnMod:
|  |  |  |  |- 📜 FRQuestion.tsx
|  |  |  |  |- 📜 Exercise.tsx
|  |  |  |  |- 📜 ClassCard.tsx
|  |  |  |  |- 📜 ImageComponent.tsx
|  |  |  |  |- 📜 ModulesSideBar.tsx
|  |  |  |  |- 📜 MCQuestion.tsx
|  |  |  |  |- 📜 LearningModulesContent.tsx
|  |  |  |- 📂 Feedback:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 feedbackApi.ts
|  |  |  |- 📂 OpenAi:
|  |  |  |  |- 📜 openAiUtils.ts
|  |  |  |- 📂 Dashboard:
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TrainDataGrid.tsx
|  |  |  |  |  |- 📜 TrainBarChart.tsx
|  |  |  |  |  |- 📜 TrainDoughnutChart.tsx
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 dashboardApi.ts
|  |  |- 📂 common:
|  |  |  |- 📂 components:
|  |  |  |  |- 📜 EmailInput.tsx
|  |  |  |  |- 📜 ClientOnlyPortal.tsx
|  |  |  |  |- 📜 Footer.tsx
|  |  |  |  |- 📜 TitleText.tsx
|  |  |  |  |- 📜 NavBarMain.tsx
|  |  |  |  |- 📜 Spacer.tsx
|  |  |  |  |- 📜 DlpTooltip.tsx
|  |  |  |  |- 📜 HtmlTooltip.tsx
|  |  |  |- 📂 utils:
|  |  |  |  |- 📜 dateFormat.ts
|  |  |  |  |- 📜 firebase.ts
|  |  |  |  |- 📜 dndHelpers.ts
|  |  |  |- 📂 styles:
|  |  |  |  |- 📜 globals.css
|  |  |  |  |- 📜 Home.module.css
|  |  |  |- 📂 redux:
|  |  |  |  |- 📜 train.ts
|  |  |  |  |- 📜 backendApi.ts
|  |  |  |  |- 📜 userLogin.ts
|  |  |  |  |- 📜 hooks.ts
|  |  |  |  |- 📜 store.ts
|  |  |- 📂 __tests__:
|  |  |  |- 📂 common:
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TitleText.test.tsx
|  |  |- 📂 backend_outputs:
|  |  |  |- 📜 my_deep_learning_model.onnx : Last ONNX file output
|  |  |  |- 📜 model.pt : Last model.pt output
|  |  |  |- 📜 model.pkl
|  |  |- 📜 next-env.d.ts
|  |  |- 📜 iris.csv : Sample CSV data
|  |  |- 📜 GlobalStyle.ts
|  |  |- 📜 constants.ts
|  |- 📜 tsconfig.json
|  |- 📜 pnpm-lock.yaml
|  |- 📜 package.json
|  |- 📜 next.config.js
|  |- 📜 jest.config.ts
|  |- 📜 next-env.d.ts
|  |- 📜 .eslintignore
|  |- 📜 .eslintrc.json
```

