# Architecture

## Training Architecture

```
📦 training
|  |- 📂 training:
|  |  |- 📂 routes:
|  |  |  |- 📂 image:
|  |  |  |  |- 📜 schemas.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |  |- 📜 image.py
|  |  |  |- 📂 datasets:
|  |  |  |  |- 📂 default:
|  |  |  |  |  |- 📜 schemas.py
|  |  |  |  |  |- 📜 columns.py
|  |  |  |  |  |- 📜 __init__.py
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
|  |  |- 📂 core:
|  |  |  |- 📜 optimizer.py : what optimizer to use (ie: SGD or Adam for now)
|  |  |  |- 📜 criterion.py
|  |  |  |- 📜 dl_model.py : torch model based on user specifications from drag and drop
|  |  |  |- 📜 dataset.py : read in the dataset through URL or file upload
|  |  |  |- 📜 authenticator.py
|  |  |  |- 📜 __init__.py
|  |  |  |- 📜 trainer.py
|  |  |- 📜 wsgi.py
|  |  |- 📜 settings.py
|  |  |- 📜 urls.py
|  |  |- 📜 asgi.py
|  |  |- 📜 __init__.py
|  |- 📜 pytest.ini
|  |- 📜 cli.py
|  |- 📜 README.md
|  |- 📜 Dockerfile.prod
|  |- 📜 pyproject.toml
|  |- 📜 docker-compose.yml
|  |- 📜 poetry.lock
|  |- 📜 Dockerfile
|  |- 📜 manage.py
|  |- 📜 docker-compose.prod.yml
|  |- 📜 environment.yml
```

## Frontend Architecture

```
📦 frontend
|  |- 📂 public:
|  |  |- 📂 images:
|  |  |  |- 📂 learn_mod_images:
|  |  |  |  |- 📜 LeakyReLUactivation.png
|  |  |  |  |- 📜 neuralnet.png
|  |  |  |  |- 📜 tanhactivation.png
|  |  |  |  |- 📜 lossExample.png
|  |  |  |  |- 📜 sigmoidfunction.png
|  |  |  |  |- 📜 neuron.png
|  |  |  |  |- 📜 ReLUactivation.png
|  |  |  |  |- 📜 sigmoidactivation.png
|  |  |  |  |- 📜 lossExampleEquation.png
|  |  |  |  |- 📜 neuronWithEquation.png
|  |  |  |  |- 📜 robotImage.jpg
|  |  |  |  |- 📜 binarystepactivation.png
|  |  |  |  |- 📜 lossExampleTable.png
|  |  |  |- 📂 wiki_images:
|  |  |  |  |- 📜 avgpool_maxpool.gif
|  |  |  |  |- 📜 tanh_equation.png
|  |  |  |  |- 📜 maxpool2d.gif
|  |  |  |  |- 📜 dropout_diagram.png
|  |  |  |  |- 📜 batchnorm_diagram.png
|  |  |  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |  |  |- 📜 tanh_plot.png
|  |  |  |  |- 📜 sigmoid_equation.png
|  |  |  |  |- 📜 conv2d.gif
|  |  |  |  |- 📜 conv2d2.gif
|  |  |  |- 📂 logos:
|  |  |  |  |- 📂 dlp_branding:
|  |  |  |  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |- 📜 pytorch-logo.png
|  |  |  |  |- 📜 dsgt-logo-dark.png
|  |  |  |  |- 📜 flask-logo.png
|  |  |  |  |- 📜 dsgt-logo-light.png
|  |  |  |  |- 📜 aws-logo.png
|  |  |  |  |- 📜 github.png
|  |  |  |  |- 📜 google.png
|  |  |  |  |- 📜 pandas-logo.png
|  |  |  |  |- 📜 python-logo.png
|  |  |  |  |- 📜 dsgt-logo-white-back.png
|  |  |  |  |- 📜 react-logo.png
|  |  |  |- 📜 demo_video.gif : GIF tutorial of a simple classification training session
|  |  |- 📜 dlp-logo.ico : DLP Logo
|  |  |- 📜 manifest.json : Default React file for choosing icon based on
|  |  |- 📜 index.html : Base HTML file that will be initially rendered
|  |  |- 📜 robots.txt
|  |- 📂 layer_docs:
|  |  |- 📜 Linear.md : Doc for Linear layer
|  |  |- 📜 Softmax.md : Doc for Softmax layer
|  |  |- 📜 ReLU.md : Doc for ReLU later
|  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |- 📂 src:
|  |  |- 📂 features:
|  |  |  |- 📂 Dashboard:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 dashboardApi.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TrainDataGrid.tsx
|  |  |  |  |  |- 📜 TrainBarChart.tsx
|  |  |  |  |  |- 📜 TrainDoughnutChart.tsx
|  |  |  |- 📂 LearnMod:
|  |  |  |  |- 📜 FRQuestion.tsx
|  |  |  |  |- 📜 MCQuestion.tsx
|  |  |  |  |- 📜 ClassCard.tsx
|  |  |  |  |- 📜 ImageComponent.tsx
|  |  |  |  |- 📜 LearningModulesContent.tsx
|  |  |  |  |- 📜 ModulesSideBar.tsx
|  |  |  |  |- 📜 Exercise.tsx
|  |  |  |- 📂 Feedback:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 feedbackApi.ts
|  |  |  |- 📂 Train:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 trainspaceApi.ts
|  |  |  |  |  |- 📜 trainspaceSlice.ts
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
|  |  |  |  |  |  |  |- 📜 TabularFlow.tsx
|  |  |  |  |  |  |  |- 📜 TabularParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 TabularDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularReviewStep.tsx
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |  |- 📂 Image:
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 imageApi.ts
|  |  |  |  |  |  |  |- 📜 imageActions.ts
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 imageConstants.ts
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 imageTypes.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 ImageReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageFlow.tsx
|  |  |  |  |  |  |  |- 📜 ImageParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageTrainspace.tsx
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |- 📂 constants:
|  |  |  |  |  |- 📜 trainConstants.ts
|  |  |  |  |- 📂 types:
|  |  |  |  |  |- 📜 trainTypes.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 CreateTrainspace.tsx
|  |  |  |  |  |- 📜 DatasetStepLayout.tsx
|  |  |  |  |  |- 📜 TrainspaceLayout.tsx
|  |  |  |- 📂 OpenAi:
|  |  |  |  |- 📜 openAiUtils.ts
|  |  |- 📂 common:
|  |  |  |- 📂 styles:
|  |  |  |  |- 📜 globals.css
|  |  |  |  |- 📜 Home.module.css
|  |  |  |- 📂 redux:
|  |  |  |  |- 📜 userLogin.ts
|  |  |  |  |- 📜 store.ts
|  |  |  |  |- 📜 hooks.ts
|  |  |  |  |- 📜 backendApi.ts
|  |  |  |  |- 📜 train.ts
|  |  |  |- 📂 utils:
|  |  |  |  |- 📜 firebase.ts
|  |  |  |  |- 📜 dndHelpers.ts
|  |  |  |  |- 📜 dateFormat.ts
|  |  |  |- 📂 components:
|  |  |  |  |- 📜 NavBarMain.tsx
|  |  |  |  |- 📜 Footer.tsx
|  |  |  |  |- 📜 ClientOnlyPortal.tsx
|  |  |  |  |- 📜 Spacer.tsx
|  |  |  |  |- 📜 EmailInput.tsx
|  |  |  |  |- 📜 TitleText.tsx
|  |  |  |  |- 📜 HtmlTooltip.tsx
|  |  |  |  |- 📜 DlpTooltip.tsx
|  |  |- 📂 backend_outputs:
|  |  |  |- 📜 model.pkl
|  |  |  |- 📜 my_deep_learning_model.onnx : Last ONNX file output
|  |  |  |- 📜 model.pt : Last model.pt output
|  |  |- 📂 pages:
|  |  |  |- 📂 train:
|  |  |  |  |- 📜 [train_space_id].tsx
|  |  |  |  |- 📜 index.tsx
|  |  |  |- 📜 login.tsx
|  |  |  |- 📜 feedback.tsx
|  |  |  |- 📜 _app.tsx
|  |  |  |- 📜 forgot.tsx
|  |  |  |- 📜 learn.tsx
|  |  |  |- 📜 LearnContent.tsx
|  |  |  |- 📜 settings.tsx
|  |  |  |- 📜 dashboard.tsx
|  |  |  |- 📜 about.tsx
|  |  |  |- 📜 wiki.tsx
|  |  |  |- 📜 _document.tsx
|  |  |- 📜 constants.ts
|  |  |- 📜 iris.csv : Sample CSV data
|  |  |- 📜 GlobalStyle.ts
|  |  |- 📜 next-env.d.ts
|  |- 📜 .eslintignore
|  |- 📜 tsconfig.json
|  |- 📜 .eslintrc.json
|  |- 📜 pnpm-lock.yaml
|  |- 📜 jest.config.js
|  |- 📜 package.json
|  |- 📜 next.config.js
|  |- 📜 next-env.d.ts
```

