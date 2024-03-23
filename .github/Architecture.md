# Architecture

## Training Architecture

```
📦 training
|  |- 📂 training:
|  |  |- 📂 core:
|  |  |  |- 📜 __init__.py
|  |  |  |- 📜 dl_model.py : torch model based on user specifications from drag and drop
|  |  |  |- 📜 criterion.py
|  |  |  |- 📜 dataset.py : read in the dataset through URL or file upload
|  |  |  |- 📜 authenticator.py
|  |  |  |- 📜 trainer.py
|  |  |  |- 📜 optimizer.py : what optimizer to use (ie: SGD or Adam for now)
|  |  |- 📂 routes:
|  |  |  |- 📂 image:
|  |  |  |  |- 📜 __init__.py
|  |  |  |  |- 📜 image.py
|  |  |  |  |- 📜 schemas.py
|  |  |  |- 📂 datasets:
|  |  |  |  |- 📂 default:
|  |  |  |  |  |- 📜 __init__.py
|  |  |  |  |  |- 📜 columns.py
|  |  |  |  |  |- 📜 schemas.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |- 📂 tabular:
|  |  |  |  |- 📜 __init__.py
|  |  |  |  |- 📜 tabular.py
|  |  |  |  |- 📜 schemas.py
|  |  |  |- 📜 __init__.py
|  |  |  |- 📜 schemas.py
|  |  |- 📂 middleware:
|  |  |  |- 📜 health_check_middleware.py
|  |  |  |- 📜 __init__.py
|  |  |- 📜 __init__.py
|  |  |- 📜 wsgi.py
|  |  |- 📜 settings.py
|  |  |- 📜 urls.py
|  |  |- 📜 asgi.py
|  |- 📜 pytest.ini
|  |- 📜 Dockerfile.prod
|  |- 📜 Dockerfile
|  |- 📜 docker-compose.prod.yml
|  |- 📜 manage.py
|  |- 📜 poetry.lock
|  |- 📜 environment.yml
|  |- 📜 cli.py
|  |- 📜 README.md
|  |- 📜 docker-compose.yml
|  |- 📜 pyproject.toml
```

## Frontend Architecture

```
📦 frontend
|  |- 📂 public:
|  |  |- 📂 images:
|  |  |  |- 📂 learn_mod_images:
|  |  |  |  |- 📜 tanhactivation.png
|  |  |  |  |- 📜 sigmoidfunction.png
|  |  |  |  |- 📜 lossExample.png
|  |  |  |  |- 📜 neuronWithEquation.png
|  |  |  |  |- 📜 binarystepactivation.png
|  |  |  |  |- 📜 robotImage.jpg
|  |  |  |  |- 📜 ReLUactivation.png
|  |  |  |  |- 📜 lossExampleTable.png
|  |  |  |  |- 📜 sigmoidactivation.png
|  |  |  |  |- 📜 LeakyReLUactivation.png
|  |  |  |  |- 📜 neuron.png
|  |  |  |  |- 📜 neuralnet.png
|  |  |  |  |- 📜 lossExampleEquation.png
|  |  |  |- 📂 wiki_images:
|  |  |  |  |- 📜 avgpool_maxpool.gif
|  |  |  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |  |  |- 📜 dropout_diagram.png
|  |  |  |  |- 📜 tanh_equation.png
|  |  |  |  |- 📜 maxpool2d.gif
|  |  |  |  |- 📜 conv2d.gif
|  |  |  |  |- 📜 tanh_plot.png
|  |  |  |  |- 📜 sigmoid_equation.png
|  |  |  |  |- 📜 batchnorm_diagram.png
|  |  |  |  |- 📜 conv2d2.gif
|  |  |  |- 📂 logos:
|  |  |  |  |- 📂 dlp_branding:
|  |  |  |  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |- 📜 pytorch-logo.png
|  |  |  |  |- 📜 dsgt-logo-dark.png
|  |  |  |  |- 📜 dsgt-logo-white-back.png
|  |  |  |  |- 📜 github.png
|  |  |  |  |- 📜 python-logo.png
|  |  |  |  |- 📜 pandas-logo.png
|  |  |  |  |- 📜 dsgt-logo-light.png
|  |  |  |  |- 📜 flask-logo.png
|  |  |  |  |- 📜 react-logo.png
|  |  |  |  |- 📜 google.png
|  |  |  |  |- 📜 aws-logo.png
|  |  |  |- 📜 demo_video.gif : GIF tutorial of a simple classification training session
|  |  |- 📜 index.html : Base HTML file that will be initially rendered
|  |  |- 📜 robots.txt
|  |  |- 📜 manifest.json : Default React file for choosing icon based on
|  |  |- 📜 dlp-logo.ico : DLP Logo
|  |- 📂 src:
|  |  |- 📂 backend_outputs:
|  |  |  |- 📜 model.pkl
|  |  |  |- 📜 my_deep_learning_model.onnx : Last ONNX file output
|  |  |  |- 📜 model.pt : Last model.pt output
|  |  |- 📂 common:
|  |  |  |- 📂 components:
|  |  |  |  |- 📜 EmailInput.tsx
|  |  |  |  |- 📜 TitleText.tsx
|  |  |  |  |- 📜 DlpTooltip.tsx
|  |  |  |  |- 📜 Spacer.tsx
|  |  |  |  |- 📜 NavBarMain.tsx
|  |  |  |  |- 📜 HtmlTooltip.tsx
|  |  |  |  |- 📜 ClientOnlyPortal.tsx
|  |  |  |  |- 📜 Footer.tsx
|  |  |  |- 📂 styles:
|  |  |  |  |- 📜 Home.module.css
|  |  |  |  |- 📜 globals.css
|  |  |  |- 📂 redux:
|  |  |  |  |- 📜 userLogin.ts
|  |  |  |  |- 📜 backendApi.ts
|  |  |  |  |- 📜 hooks.ts
|  |  |  |  |- 📜 store.ts
|  |  |  |  |- 📜 train.ts
|  |  |  |- 📂 utils:
|  |  |  |  |- 📜 dateFormat.ts
|  |  |  |  |- 📜 firebase.ts
|  |  |  |  |- 📜 dndHelpers.ts
|  |  |- 📂 features:
|  |  |  |- 📂 OpenAi:
|  |  |  |  |- 📜 openAiUtils.ts
|  |  |  |- 📂 LearnMod:
|  |  |  |  |- 📜 Exercise.tsx
|  |  |  |  |- 📜 FRQuestion.tsx
|  |  |  |  |- 📜 ClassCard.tsx
|  |  |  |  |- 📜 ModulesSideBar.tsx
|  |  |  |  |- 📜 MCQuestion.tsx
|  |  |  |  |- 📜 LearningModulesContent.tsx
|  |  |  |  |- 📜 ImageComponent.tsx
|  |  |  |- 📂 Train:
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TrainspaceLayout.tsx
|  |  |  |  |  |- 📜 CreateTrainspace.tsx
|  |  |  |  |  |- 📜 DatasetStepLayout.tsx
|  |  |  |  |- 📂 constants:
|  |  |  |  |  |- 📜 trainConstants.ts
|  |  |  |  |- 📂 types:
|  |  |  |  |  |- 📜 trainTypes.ts
|  |  |  |  |- 📂 features:
|  |  |  |  |  |- 📂 Tabular:
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 TabularTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 TabularDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularFlow.tsx
|  |  |  |  |  |  |  |- 📜 TabularReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularParametersStep.tsx
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 tabularConstants.ts
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 tabularTypes.ts
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 tabularActions.ts
|  |  |  |  |  |  |  |- 📜 tabularApi.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |  |- 📂 Image:
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 ImageReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 ImageFlow.tsx
|  |  |  |  |  |  |  |- 📜 ImageDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageParametersStep.tsx
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 imageConstants.ts
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 imageTypes.ts
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 imageApi.ts
|  |  |  |  |  |  |  |- 📜 imageActions.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 trainspaceApi.ts
|  |  |  |  |  |- 📜 trainspaceSlice.ts
|  |  |  |- 📂 Feedback:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 feedbackApi.ts
|  |  |  |- 📂 Dashboard:
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TrainDoughnutChart.tsx
|  |  |  |  |  |- 📜 TrainBarChart.tsx
|  |  |  |  |  |- 📜 TrainDataGrid.tsx
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 dashboardApi.ts
|  |  |- 📂 __tests__:
|  |  |  |- 📂 common:
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TitleText.test.tsx
|  |  |- 📂 pages:
|  |  |  |- 📂 train:
|  |  |  |  |- 📜 [train_space_id].tsx
|  |  |  |  |- 📜 index.tsx
|  |  |  |- 📜 settings.tsx
|  |  |  |- 📜 _app.tsx
|  |  |  |- 📜 feedback.tsx
|  |  |  |- 📜 _document.tsx
|  |  |  |- 📜 wiki.tsx
|  |  |  |- 📜 login.tsx
|  |  |  |- 📜 learn.tsx
|  |  |  |- 📜 forgot.tsx
|  |  |  |- 📜 LearnContent.tsx
|  |  |  |- 📜 dashboard.tsx
|  |  |  |- 📜 about.tsx
|  |  |- 📜 constants.ts
|  |  |- 📜 next-env.d.ts
|  |  |- 📜 iris.csv : Sample CSV data
|  |  |- 📜 GlobalStyle.ts
|  |- 📂 layer_docs:
|  |  |- 📜 Softmax.md : Doc for Softmax layer
|  |  |- 📜 Linear.md : Doc for Linear layer
|  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |- 📜 ReLU.md : Doc for ReLU later
|  |- 📜 .eslintignore
|  |- 📜 next.config.js
|  |- 📜 next-env.d.ts
|  |- 📜 pnpm-lock.yaml
|  |- 📜 tsconfig.json
|  |- 📜 .eslintrc.json
|  |- 📜 package.json
|  |- 📜 jest.config.ts
```

