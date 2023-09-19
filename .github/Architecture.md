# Architecture

## Backend Architecture

```
📦 backend
|  |- 📂 dl:
|  |  |- 📜 dl_model_parser.py : parse the user specified pytorch model
|  |  |- 📜 dl_trainer.py : train a deep learning model on the dataset
|  |  |- 📜 __init__.py
|  |  |- 📜 dl_model.py : torch model based on user specifications from drag and drop
|  |  |- 📜 detection.py
|  |  |- 📜 dl_eval.py : Evaluation functions for deep learning models in Pytorch (eg: accuracy, loss, etc)
|  |- 📂 ml:
|  |  |- 📜 ml_trainer.py : train a classical machine learning learning model on the dataset
|  |  |- 📜 ml_model_parser.py
|  |  |- 📜 __init__.py
|  |- 📂 common:
|  |  |- 📜 optimizer.py : what optimizer to use (ie: SGD or Adam for now)
|  |  |- 📜 dataset.py : read in the dataset through URL or file upload
|  |  |- 📜 loss_functions.py : loss function enum
|  |  |- 📜 __init__.py
|  |  |- 📜 preprocessing.py
|  |  |- 📜 constants.py : list of helpful constants
|  |  |- 📜 ai_drive.py
|  |  |- 📜 default_datasets.py : store logic to load in default datasets from scikit-learn
|  |  |- 📜 utils.py : utility functions that could be helpful
|  |  |- 📜 email_notifier.py : Endpoint to send email notification of training results via API Gateway + AWS SES
|  |  |- 📜 kernel.py
|  |- 📂 aws_helpers:
|  |  |- 📂 dynamo_db_utils:
|  |  |  |- 📜 DynamoUnitTests.md
|  |  |  |- 📜 dynamo_db_utils.py
|  |  |  |- 📜 constants.py : list of helpful constants
|  |  |  |- 📜 trainspace_db.py
|  |  |  |- 📜 userprogress_db.py
|  |  |- 📜 __init__.py
|  |- 📜 epoch_times.csv
|  |- 📜 app.py : run the backend (entrypoint script)
|  |- 📜 __init__.py
|  |- 📜 pyproject.toml
|  |- 📜 data.csv : data csv file for use in the playground
|  |- 📜 poetry.lock
|  |- 📜 middleware.py
```

## Frontend Architecture

```
📦 frontend
|  |- 📂 src:
|  |  |- 📂 features:
|  |  |  |- 📂 Train:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 trainspaceSlice.ts
|  |  |  |  |  |- 📜 trainspaceApi.ts
|  |  |  |  |- 📂 features:
|  |  |  |  |  |- 📂 Tabular:
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 tabularApi.ts
|  |  |  |  |  |  |  |- 📜 tabularActions.ts
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 tabularConstants.ts
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 tabularTypes.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 TabularReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 TabularFlow.tsx
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
|  |  |  |  |  |  |  |- 📜 ImageParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageFlow.tsx
|  |  |  |  |  |  |  |- 📜 ImageTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 ImageReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageDatasetStep.tsx
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |- 📂 constants:
|  |  |  |  |  |- 📜 trainConstants.ts
|  |  |  |  |- 📂 types:
|  |  |  |  |  |- 📜 trainTypes.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TrainspaceLayout.tsx
|  |  |  |  |  |- 📜 DatasetStepLayout.tsx
|  |  |  |  |  |- 📜 CreateTrainspace.tsx
|  |  |  |- 📂 Feedback:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 feedbackApi.ts
|  |  |  |- 📂 OpenAi:
|  |  |  |  |- 📜 openAiUtils.ts
|  |  |  |- 📂 LearnMod:
|  |  |  |  |- 📜 ModulesSideBar.tsx
|  |  |  |  |- 📜 ImageComponent.tsx
|  |  |  |  |- 📜 MCQuestion.tsx
|  |  |  |  |- 📜 ClassCard.tsx
|  |  |  |  |- 📜 FRQuestion.tsx
|  |  |  |  |- 📜 Exercise.tsx
|  |  |  |  |- 📜 LearningModulesContent.tsx
|  |  |  |- 📂 Dashboard:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 dashboardApi.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TrainDoughnutChart.tsx
|  |  |  |  |  |- 📜 TrainDataGrid.tsx
|  |  |  |  |  |- 📜 TrainBarChart.tsx
|  |  |- 📂 common:
|  |  |  |- 📂 redux:
|  |  |  |  |- 📜 backendApi.ts
|  |  |  |  |- 📜 userLogin.ts
|  |  |  |  |- 📜 train.ts
|  |  |  |  |- 📜 store.ts
|  |  |  |  |- 📜 hooks.ts
|  |  |  |- 📂 styles:
|  |  |  |  |- 📜 globals.css
|  |  |  |  |- 📜 Home.module.css
|  |  |  |- 📂 components:
|  |  |  |  |- 📜 ClientOnlyPortal.tsx
|  |  |  |  |- 📜 EmailInput.tsx
|  |  |  |  |- 📜 NavBarMain.tsx
|  |  |  |  |- 📜 Spacer.tsx
|  |  |  |  |- 📜 HtmlTooltip.tsx
|  |  |  |  |- 📜 TitleText.tsx
|  |  |  |  |- 📜 DlpTooltip.tsx
|  |  |  |  |- 📜 Footer.tsx
|  |  |  |- 📂 utils:
|  |  |  |  |- 📜 dndHelpers.ts
|  |  |  |  |- 📜 firebase.ts
|  |  |  |  |- 📜 dateFormat.ts
|  |  |- 📂 pages:
|  |  |  |- 📂 train:
|  |  |  |  |- 📜 [train_space_id].tsx
|  |  |  |  |- 📜 index.tsx
|  |  |  |- 📜 _document.tsx
|  |  |  |- 📜 dashboard.tsx
|  |  |  |- 📜 _app.tsx
|  |  |  |- 📜 LearnContent.tsx
|  |  |  |- 📜 forgot.tsx
|  |  |  |- 📜 learn.tsx
|  |  |  |- 📜 settings.tsx
|  |  |  |- 📜 feedback.tsx
|  |  |  |- 📜 about.tsx
|  |  |  |- 📜 login.tsx
|  |  |  |- 📜 wiki.tsx
|  |  |- 📂 backend_outputs:
|  |  |  |- 📜 model.pkl
|  |  |  |- 📜 model.pt : Last model.pt output
|  |  |  |- 📜 my_deep_learning_model.onnx : Last ONNX file output
|  |  |- 📜 iris.csv : Sample CSV data
|  |  |- 📜 constants.ts
|  |  |- 📜 GlobalStyle.ts
|  |  |- 📜 next-env.d.ts
|  |- 📂 public:
|  |  |- 📂 images:
|  |  |  |- 📂 wiki_images:
|  |  |  |  |- 📜 maxpool2d.gif
|  |  |  |  |- 📜 tanh_plot.png
|  |  |  |  |- 📜 avgpool_maxpool.gif
|  |  |  |  |- 📜 batchnorm_diagram.png
|  |  |  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |  |  |- 📜 conv2d.gif
|  |  |  |  |- 📜 conv2d2.gif
|  |  |  |  |- 📜 tanh_equation.png
|  |  |  |  |- 📜 dropout_diagram.png
|  |  |  |  |- 📜 sigmoid_equation.png
|  |  |  |- 📂 learn_mod_images:
|  |  |  |  |- 📜 ReLUactivation.png
|  |  |  |  |- 📜 binarystepactivation.png
|  |  |  |  |- 📜 lossExampleEquation.png
|  |  |  |  |- 📜 sigmoidactivation.png
|  |  |  |  |- 📜 sigmoidfunction.png
|  |  |  |  |- 📜 robotImage.jpg
|  |  |  |  |- 📜 lossExample.png
|  |  |  |  |- 📜 lossExampleTable.png
|  |  |  |  |- 📜 neuronWithEquation.png
|  |  |  |  |- 📜 LeakyReLUactivation.png
|  |  |  |  |- 📜 neuron.png
|  |  |  |  |- 📜 neuralnet.png
|  |  |  |  |- 📜 tanhactivation.png
|  |  |  |- 📂 logos:
|  |  |  |  |- 📂 dlp_branding:
|  |  |  |  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |- 📜 dsgt-logo-dark.png
|  |  |  |  |- 📜 github.png
|  |  |  |  |- 📜 google.png
|  |  |  |  |- 📜 python-logo.png
|  |  |  |  |- 📜 dsgt-logo-white-back.png
|  |  |  |  |- 📜 pandas-logo.png
|  |  |  |  |- 📜 aws-logo.png
|  |  |  |  |- 📜 flask-logo.png
|  |  |  |  |- 📜 react-logo.png
|  |  |  |  |- 📜 dsgt-logo-light.png
|  |  |  |  |- 📜 pytorch-logo.png
|  |  |  |- 📜 demo_video.gif : GIF tutorial of a simple classification training session
|  |  |- 📜 dlp-logo.ico : DLP Logo
|  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |- 📜 index.html : Base HTML file that will be initially rendered
|  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |- 📜 manifest.json : Default React file for choosing icon based on
|  |  |- 📜 robots.txt
|  |- 📂 layer_docs:
|  |  |- 📜 Softmax.md : Doc for Softmax layer
|  |  |- 📜 Linear.md : Doc for Linear layer
|  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |- 📜 ReLU.md : Doc for ReLU later
|  |- 📜 .eslintrc.json
|  |- 📜 .eslintignore
|  |- 📜 package.json
|  |- 📜 jest.config.js
|  |- 📜 tsconfig.json
|  |- 📜 next.config.js
|  |- 📜 yarn.lock
|  |- 📜 next-env.d.ts
```

