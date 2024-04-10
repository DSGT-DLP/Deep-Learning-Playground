# Architecture

## Training Architecture

```
📦 training
|  |- 📂 training:
|  |  |- 📂 routes:
|  |  |  |- 📂 tabular:
|  |  |  |  |- 📜 schemas.py
|  |  |  |  |- 📜 __init__.py
|  |  |  |  |- 📜 tabular.py
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
|  |  |  |- 📜 schemas.py
|  |  |  |- 📜 __init__.py
|  |  |- 📂 core:
|  |  |  |- 📜 criterion.py
|  |  |  |- 📜 dl_model.py : torch model based on user specifications from drag and drop
|  |  |  |- 📜 dataset.py : read in the dataset through URL or file upload
|  |  |  |- 📜 __init__.py
|  |  |  |- 📜 authenticator.py
|  |  |  |- 📜 trainer.py
|  |  |  |- 📜 optimizer.py : what optimizer to use (ie: SGD or Adam for now)
|  |  |- 📂 middleware:
|  |  |  |- 📜 __init__.py
|  |  |  |- 📜 health_check_middleware.py
|  |  |- 📜 settings.py
|  |  |- 📜 urls.py
|  |  |- 📜 __init__.py
|  |  |- 📜 wsgi.py
|  |  |- 📜 asgi.py
|  |- 📜 docker-compose.yml
|  |- 📜 docker-compose.prod.yml
|  |- 📜 pyproject.toml
|  |- 📜 README.md
|  |- 📜 poetry.lock
|  |- 📜 cli.py
|  |- 📜 environment.yml
|  |- 📜 Dockerfile.prod
|  |- 📜 Dockerfile
|  |- 📜 manage.py
|  |- 📜 pytest.ini
```

## Frontend Architecture

```
📦 frontend
|  |- 📂 layer_docs:
|  |  |- 📜 Linear.md : Doc for Linear layer
|  |  |- 📜 Softmax.md : Doc for Softmax layer
|  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |- 📜 ReLU.md : Doc for ReLU later
|  |- 📂 public:
|  |  |- 📂 images:
|  |  |  |- 📂 wiki_images:
|  |  |  |  |- 📜 maxpool2d.gif
|  |  |  |  |- 📜 conv2d.gif
|  |  |  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |  |  |- 📜 tanh_equation.png
|  |  |  |  |- 📜 dropout_diagram.png
|  |  |  |  |- 📜 batchnorm_diagram.png
|  |  |  |  |- 📜 tanh_plot.png
|  |  |  |  |- 📜 conv2d2.gif
|  |  |  |  |- 📜 sigmoid_equation.png
|  |  |  |  |- 📜 avgpool_maxpool.gif
|  |  |  |- 📂 learn_mod_images:
|  |  |  |  |- 📜 lossExampleEquation.png
|  |  |  |  |- 📜 sigmoidactivation.png
|  |  |  |  |- 📜 neuralnet.png
|  |  |  |  |- 📜 binarystepactivation.png
|  |  |  |  |- 📜 lossExample.png
|  |  |  |  |- 📜 LeakyReLUactivation.png
|  |  |  |  |- 📜 lossExampleTable.png
|  |  |  |  |- 📜 tanhactivation.png
|  |  |  |  |- 📜 robotImage.jpg
|  |  |  |  |- 📜 ReLUactivation.png
|  |  |  |  |- 📜 neuron.png
|  |  |  |  |- 📜 neuronWithEquation.png
|  |  |  |  |- 📜 sigmoidfunction.png
|  |  |  |- 📂 logos:
|  |  |  |  |- 📂 dlp_branding:
|  |  |  |  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |- 📜 google.png
|  |  |  |  |- 📜 pytorch-logo.png
|  |  |  |  |- 📜 python-logo.png
|  |  |  |  |- 📜 dsgt-logo-dark.png
|  |  |  |  |- 📜 aws-logo.png
|  |  |  |  |- 📜 github.png
|  |  |  |  |- 📜 pandas-logo.png
|  |  |  |  |- 📜 react-logo.png
|  |  |  |  |- 📜 dsgt-logo-white-back.png
|  |  |  |  |- 📜 flask-logo.png
|  |  |  |  |- 📜 dsgt-logo-light.png
|  |  |  |- 📜 demo_video.gif : GIF tutorial of a simple classification training session
|  |  |- 📜 robots.txt
|  |  |- 📜 manifest.json : Default React file for choosing icon based on
|  |  |- 📜 index.html : Base HTML file that will be initially rendered
|  |  |- 📜 dlp-logo.ico : DLP Logo
|  |- 📂 src:
|  |  |- 📂 pages:
|  |  |  |- 📂 train:
|  |  |  |  |- 📜 [train_space_id].tsx
|  |  |  |  |- 📜 index.tsx
|  |  |  |- 📜 dashboard.tsx
|  |  |  |- 📜 learn.tsx
|  |  |  |- 📜 settings.tsx
|  |  |  |- 📜 about.tsx
|  |  |  |- 📜 feedback.tsx
|  |  |  |- 📜 wiki.tsx
|  |  |  |- 📜 forgot.tsx
|  |  |  |- 📜 _document.tsx
|  |  |  |- 📜 LearnContent.tsx
|  |  |  |- 📜 _app.tsx
|  |  |  |- 📜 login.tsx
|  |  |- 📂 common:
|  |  |  |- 📂 utils:
|  |  |  |  |- 📜 dateFormat.ts
|  |  |  |  |- 📜 dndHelpers.ts
|  |  |  |  |- 📜 firebase.ts
|  |  |  |- 📂 components:
|  |  |  |  |- 📜 Spacer.tsx
|  |  |  |  |- 📜 DlpTooltip.tsx
|  |  |  |  |- 📜 TitleText.tsx
|  |  |  |  |- 📜 EmailInput.tsx
|  |  |  |  |- 📜 HtmlTooltip.tsx
|  |  |  |  |- 📜 ClientOnlyPortal.tsx
|  |  |  |  |- 📜 Footer.tsx
|  |  |  |  |- 📜 NavBarMain.tsx
|  |  |  |- 📂 styles:
|  |  |  |  |- 📜 Home.module.css
|  |  |  |  |- 📜 globals.css
|  |  |  |- 📂 redux:
|  |  |  |  |- 📜 hooks.ts
|  |  |  |  |- 📜 store.ts
|  |  |  |  |- 📜 backendApi.ts
|  |  |  |  |- 📜 userLogin.ts
|  |  |  |  |- 📜 train.ts
|  |  |- 📂 features:
|  |  |  |- 📂 Train:
|  |  |  |  |- 📂 constants:
|  |  |  |  |  |- 📜 trainConstants.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 CreateTrainspace.tsx
|  |  |  |  |  |- 📜 TrainspaceLayout.tsx
|  |  |  |  |  |- 📜 DatasetStepLayout.tsx
|  |  |  |  |- 📂 features:
|  |  |  |  |  |- 📂 Image:
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 imageConstants.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 ImageTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 ImageParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageFlow.tsx
|  |  |  |  |  |  |  |- 📜 ImageDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageReviewStep.tsx
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 imageActions.ts
|  |  |  |  |  |  |  |- 📜 imageApi.ts
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 imageTypes.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |  |- 📂 Tabular:
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 tabularConstants.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 TabularDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularFlow.tsx
|  |  |  |  |  |  |  |- 📜 TabularTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 TabularParametersStep.tsx
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 tabularApi.ts
|  |  |  |  |  |  |  |- 📜 tabularActions.ts
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 tabularTypes.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 trainspaceApi.ts
|  |  |  |  |  |- 📜 trainspaceSlice.ts
|  |  |  |  |- 📂 types:
|  |  |  |  |  |- 📜 trainTypes.ts
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
|  |  |  |- 📂 LearnMod:
|  |  |  |  |- 📜 FRQuestion.tsx
|  |  |  |  |- 📜 MCQuestion.tsx
|  |  |  |  |- 📜 LearningModulesContent.tsx
|  |  |  |  |- 📜 Exercise.tsx
|  |  |  |  |- 📜 ClassCard.tsx
|  |  |  |  |- 📜 ImageComponent.tsx
|  |  |  |  |- 📜 ModulesSideBar.tsx
|  |  |  |- 📂 OpenAi:
|  |  |  |  |- 📜 openAiUtils.ts
|  |  |- 📂 backend_outputs:
|  |  |  |- 📜 my_deep_learning_model.onnx : Last ONNX file output
|  |  |  |- 📜 model.pkl
|  |  |  |- 📜 model.pt : Last model.pt output
|  |  |- 📂 __tests__:
|  |  |  |- 📂 common:
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TitleText.test.tsx
|  |  |- 📜 next-env.d.ts
|  |  |- 📜 iris.csv : Sample CSV data
|  |  |- 📜 GlobalStyle.ts
|  |  |- 📜 constants.ts
|  |- 📜 next-env.d.ts
|  |- 📜 .eslintignore
|  |- 📜 jest.config.ts
|  |- 📜 pnpm-lock.yaml
|  |- 📜 .eslintrc.json
|  |- 📜 next.config.js
|  |- 📜 tsconfig.json
|  |- 📜 package.json
```

