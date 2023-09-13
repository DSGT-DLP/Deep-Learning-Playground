# Architecture

## Backend Architecture

```
📦 backend
|  |- 📂 aws_helpers:
|  |  |- 📂 dynamo_db_utils:
|  |  |  |- 📜 DynamoUnitTests.md
|  |  |  |- 📜 dynamo_db_utils.py
|  |  |  |- 📜 trainspace_db.py
|  |  |  |- 📜 constants.py : list of helpful constants
|  |  |  |- 📜 userprogress_db.py
|  |  |- 📜 __init__.py
|  |- 📂 ml:
|  |  |- 📜 ml_trainer.py : train a classical machine learning learning model on the dataset
|  |  |- 📜 __init__.py
|  |  |- 📜 ml_model_parser.py
|  |- 📂 dl:
|  |  |- 📜 __init__.py
|  |  |- 📜 dl_trainer.py : train a deep learning model on the dataset
|  |  |- 📜 dl_model.py : torch model based on user specifications from drag and drop
|  |  |- 📜 dl_model_parser.py : parse the user specified pytorch model
|  |  |- 📜 detection.py
|  |  |- 📜 dl_eval.py : Evaluation functions for deep learning models in Pytorch (eg: accuracy, loss, etc)
|  |- 📂 common:
|  |  |- 📜 default_datasets.py : store logic to load in default datasets from scikit-learn
|  |  |- 📜 __init__.py
|  |  |- 📜 preprocessing.py
|  |  |- 📜 dataset.py : read in the dataset through URL or file upload
|  |  |- 📜 constants.py : list of helpful constants
|  |  |- 📜 optimizer.py : what optimizer to use (ie: SGD or Adam for now)
|  |  |- 📜 ai_drive.py
|  |  |- 📜 email_notifier.py : Endpoint to send email notification of training results via API Gateway + AWS SES
|  |  |- 📜 loss_functions.py : loss function enum
|  |  |- 📜 kernel.py
|  |  |- 📜 utils.py : utility functions that could be helpful
|  |- 📜 __init__.py
|  |- 📜 middleware.py
|  |- 📜 poetry.lock
|  |- 📜 epoch_times.csv
|  |- 📜 app.py : run the backend (entrypoint script)
|  |- 📜 data.csv : data csv file for use in the playground
|  |- 📜 pyproject.toml
```

## Frontend Architecture

```
📦 frontend
|  |- 📂 src:
|  |  |- 📂 backend_outputs:
|  |  |  |- 📜 model.pt : Last model.pt output
|  |  |  |- 📜 my_deep_learning_model.onnx : Last ONNX file output
|  |  |  |- 📜 model.pkl
|  |  |- 📂 pages:
|  |  |  |- 📂 train:
|  |  |  |  |- 📜 [train_space_id].tsx
|  |  |  |  |- 📜 index.tsx
|  |  |  |- 📜 feedback.tsx
|  |  |  |- 📜 settings.tsx
|  |  |  |- 📜 _app.tsx
|  |  |  |- 📜 learn.tsx
|  |  |  |- 📜 login.tsx
|  |  |  |- 📜 wiki.tsx
|  |  |  |- 📜 _document.tsx
|  |  |  |- 📜 about.tsx
|  |  |  |- 📜 forgot.tsx
|  |  |  |- 📜 dashboard.tsx
|  |  |  |- 📜 LearnContent.tsx
|  |  |- 📂 features:
|  |  |  |- 📂 Train:
|  |  |  |  |- 📂 constants:
|  |  |  |  |  |- 📜 trainConstants.ts
|  |  |  |  |- 📂 features:
|  |  |  |  |  |- 📂 Image:
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 imageConstants.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 ImageFlow.tsx
|  |  |  |  |  |  |  |- 📜 ImageTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 ImageDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageParametersStep.tsx
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 imageTypes.ts
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 imageActions.ts
|  |  |  |  |  |  |  |- 📜 imageApi.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |  |- 📂 Tabular:
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 tabularConstants.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 TabularParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularFlow.tsx
|  |  |  |  |  |  |  |- 📜 TabularTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 TabularReviewStep.tsx
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 tabularTypes.ts
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 tabularActions.ts
|  |  |  |  |  |  |  |- 📜 tabularApi.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 CreateTrainspace.tsx
|  |  |  |  |  |- 📜 DatasetStepLayout.tsx
|  |  |  |  |  |- 📜 TrainspaceLayout.tsx
|  |  |  |  |- 📂 types:
|  |  |  |  |  |- 📜 trainTypes.ts
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 trainspaceApi.ts
|  |  |  |  |  |- 📜 trainspaceSlice.ts
|  |  |  |- 📂 Dashboard:
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TrainDoughnutChart.tsx
|  |  |  |  |  |- 📜 TrainBarChart.tsx
|  |  |  |  |  |- 📜 TrainDataGrid.tsx
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 dashboardApi.ts
|  |  |  |- 📂 LearnMod:
|  |  |  |  |- 📜 LearningModulesContent.tsx
|  |  |  |  |- 📜 Exercise.tsx
|  |  |  |  |- 📜 ImageComponent.tsx
|  |  |  |  |- 📜 ModulesSideBar.tsx
|  |  |  |  |- 📜 MCQuestion.tsx
|  |  |  |  |- 📜 FRQuestion.tsx
|  |  |  |  |- 📜 ClassCard.tsx
|  |  |  |- 📂 Feedback:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 feedbackApi.ts
|  |  |  |- 📂 OpenAi:
|  |  |  |  |- 📜 openAiUtils.ts
|  |  |- 📂 common:
|  |  |  |- 📂 styles:
|  |  |  |  |- 📜 globals.css
|  |  |  |  |- 📜 Home.module.css
|  |  |  |- 📂 components:
|  |  |  |  |- 📜 ClientOnlyPortal.tsx
|  |  |  |  |- 📜 EmailInput.tsx
|  |  |  |  |- 📜 HtmlTooltip.tsx
|  |  |  |  |- 📜 TitleText.tsx
|  |  |  |  |- 📜 NavBarMain.tsx
|  |  |  |  |- 📜 Footer.tsx
|  |  |  |  |- 📜 DlpTooltip.tsx
|  |  |  |  |- 📜 Spacer.tsx
|  |  |  |- 📂 utils:
|  |  |  |  |- 📜 firebase.ts
|  |  |  |  |- 📜 dateFormat.ts
|  |  |  |  |- 📜 dndHelpers.ts
|  |  |  |- 📂 redux:
|  |  |  |  |- 📜 backendApi.ts
|  |  |  |  |- 📜 userLogin.ts
|  |  |  |  |- 📜 hooks.ts
|  |  |  |  |- 📜 store.ts
|  |  |  |  |- 📜 train.ts
|  |  |- 📜 iris.csv : Sample CSV data
|  |  |- 📜 next-env.d.ts
|  |  |- 📜 constants.ts
|  |  |- 📜 GlobalStyle.ts
|  |- 📂 layer_docs:
|  |  |- 📜 Linear.md : Doc for Linear layer
|  |  |- 📜 Softmax.md : Doc for Softmax layer
|  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |- 📜 ReLU.md : Doc for ReLU later
|  |- 📂 public:
|  |  |- 📂 images:
|  |  |  |- 📂 logos:
|  |  |  |  |- 📂 dlp_branding:
|  |  |  |  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |- 📜 dsgt-logo-dark.png
|  |  |  |  |- 📜 github.png
|  |  |  |  |- 📜 dsgt-logo-light.png
|  |  |  |  |- 📜 pandas-logo.png
|  |  |  |  |- 📜 react-logo.png
|  |  |  |  |- 📜 python-logo.png
|  |  |  |  |- 📜 flask-logo.png
|  |  |  |  |- 📜 pytorch-logo.png
|  |  |  |  |- 📜 aws-logo.png
|  |  |  |  |- 📜 google.png
|  |  |  |  |- 📜 dsgt-logo-white-back.png
|  |  |  |- 📂 learn_mod_images:
|  |  |  |  |- 📜 lossExampleTable.png
|  |  |  |  |- 📜 tanhactivation.png
|  |  |  |  |- 📜 sigmoidactivation.png
|  |  |  |  |- 📜 LeakyReLUactivation.png
|  |  |  |  |- 📜 ReLUactivation.png
|  |  |  |  |- 📜 robotImage.jpg
|  |  |  |  |- 📜 lossExample.png
|  |  |  |  |- 📜 binarystepactivation.png
|  |  |  |  |- 📜 lossExampleEquation.png
|  |  |  |  |- 📜 neuronWithEquation.png
|  |  |  |  |- 📜 neuron.png
|  |  |  |  |- 📜 neuralnet.png
|  |  |  |  |- 📜 sigmoidfunction.png
|  |  |  |- 📂 wiki_images:
|  |  |  |  |- 📜 avgpool_maxpool.gif
|  |  |  |  |- 📜 conv2d2.gif
|  |  |  |  |- 📜 dropout_diagram.png
|  |  |  |  |- 📜 batchnorm_diagram.png
|  |  |  |  |- 📜 tanh_plot.png
|  |  |  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |  |  |- 📜 maxpool2d.gif
|  |  |  |  |- 📜 tanh_equation.png
|  |  |  |  |- 📜 conv2d.gif
|  |  |  |  |- 📜 sigmoid_equation.png
|  |  |  |- 📜 demo_video.gif : GIF tutorial of a simple classification training session
|  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |- 📜 robots.txt
|  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |- 📜 manifest.json : Default React file for choosing icon based on
|  |  |- 📜 index.html : Base HTML file that will be initially rendered
|  |  |- 📜 dlp-logo.ico : DLP Logo
|  |- 📜 package.json
|  |- 📜 jest.config.js
|  |- 📜 .eslintrc.json
|  |- 📜 tsconfig.json
|  |- 📜 yarn.lock
|  |- 📜 .eslintignore
|  |- 📜 next-env.d.ts
|  |- 📜 next.config.js
```

