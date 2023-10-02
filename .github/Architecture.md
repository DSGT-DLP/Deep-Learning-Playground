# Architecture

## Backend Architecture

```
📦 backend
|  |- 📂 aws_helpers:
|  |  |- 📂 dynamo_db_utils:
|  |  |  |- 📜 trainspace_db.py
|  |  |  |- 📜 constants.py : list of helpful constants
|  |  |  |- 📜 userprogress_db.py
|  |  |  |- 📜 DynamoUnitTests.md
|  |  |  |- 📜 dynamo_db_utils.py
|  |  |- 📜 __init__.py
|  |- 📂 ml:
|  |  |- 📜 ml_model_parser.py
|  |  |- 📜 __init__.py
|  |  |- 📜 ml_trainer.py : train a classical machine learning learning model on the dataset
|  |- 📂 common:
|  |  |- 📜 loss_functions.py : loss function enum
|  |  |- 📜 default_datasets.py : store logic to load in default datasets from scikit-learn
|  |  |- 📜 dataset.py : read in the dataset through URL or file upload
|  |  |- 📜 constants.py : list of helpful constants
|  |  |- 📜 utils.py : utility functions that could be helpful
|  |  |- 📜 __init__.py
|  |  |- 📜 preprocessing.py
|  |  |- 📜 ai_drive.py
|  |  |- 📜 kernel.py
|  |  |- 📜 email_notifier.py : Endpoint to send email notification of training results via API Gateway + AWS SES
|  |  |- 📜 optimizer.py : what optimizer to use (ie: SGD or Adam for now)
|  |- 📂 dl:
|  |  |- 📜 dl_eval.py : Evaluation functions for deep learning models in Pytorch (eg: accuracy, loss, etc)
|  |  |- 📜 dl_model.py : torch model based on user specifications from drag and drop
|  |  |- 📜 __init__.py
|  |  |- 📜 dl_trainer.py : train a deep learning model on the dataset
|  |  |- 📜 dl_model_parser.py : parse the user specified pytorch model
|  |  |- 📜 detection.py
|  |- 📜 app.py : run the backend (entrypoint script)
|  |- 📜 poetry.lock
|  |- 📜 epoch_times.csv
|  |- 📜 __init__.py
|  |- 📜 data.csv : data csv file for use in the playground
|  |- 📜 middleware.py
|  |- 📜 pyproject.toml
```

## Frontend Architecture

```
📦 frontend
|  |- 📂 layer_docs:
|  |  |- 📜 Softmax.md : Doc for Softmax layer
|  |  |- 📜 Linear.md : Doc for Linear layer
|  |  |- 📜 ReLU.md : Doc for ReLU later
|  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |- 📂 src:
|  |  |- 📂 features:
|  |  |  |- 📂 LearnMod:
|  |  |  |  |- 📜 ModulesSideBar.tsx
|  |  |  |  |- 📜 MCQuestion.tsx
|  |  |  |  |- 📜 ClassCard.tsx
|  |  |  |  |- 📜 ImageComponent.tsx
|  |  |  |  |- 📜 FRQuestion.tsx
|  |  |  |  |- 📜 LearningModulesContent.tsx
|  |  |  |  |- 📜 Exercise.tsx
|  |  |  |- 📂 Dashboard:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 dashboardApi.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TrainBarChart.tsx
|  |  |  |  |  |- 📜 TrainDoughnutChart.tsx
|  |  |  |  |  |- 📜 TrainDataGrid.tsx
|  |  |  |- 📂 Train:
|  |  |  |  |- 📂 features:
|  |  |  |  |  |- 📂 Tabular:
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 tabularActions.ts
|  |  |  |  |  |  |  |- 📜 tabularApi.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 TabularReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularFlow.tsx
|  |  |  |  |  |  |  |- 📜 TabularTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 TabularParametersStep.tsx
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 tabularTypes.ts
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 tabularConstants.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |  |- 📂 Image:
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 imageActions.ts
|  |  |  |  |  |  |  |- 📜 imageApi.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 ImageDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 ImageReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageFlow.tsx
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 imageTypes.ts
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 imageConstants.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 trainspaceApi.ts
|  |  |  |  |  |- 📜 trainspaceSlice.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 DatasetStepLayout.tsx
|  |  |  |  |  |- 📜 CreateTrainspace.tsx
|  |  |  |  |  |- 📜 TrainspaceLayout.tsx
|  |  |  |  |- 📂 types:
|  |  |  |  |  |- 📜 trainTypes.ts
|  |  |  |  |- 📂 constants:
|  |  |  |  |  |- 📜 trainConstants.ts
|  |  |  |- 📂 Feedback:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 feedbackApi.ts
|  |  |  |- 📂 OpenAi:
|  |  |  |  |- 📜 openAiUtils.ts
|  |  |- 📂 pages:
|  |  |  |- 📂 train:
|  |  |  |  |- 📜 [train_space_id].tsx
|  |  |  |  |- 📜 index.tsx
|  |  |  |- 📜 LearnContent.tsx
|  |  |  |- 📜 feedback.tsx
|  |  |  |- 📜 dashboard.tsx
|  |  |  |- 📜 learn.tsx
|  |  |  |- 📜 wiki.tsx
|  |  |  |- 📜 forgot.tsx
|  |  |  |- 📜 about.tsx
|  |  |  |- 📜 _app.tsx
|  |  |  |- 📜 login.tsx
|  |  |  |- 📜 _document.tsx
|  |  |  |- 📜 settings.tsx
|  |  |- 📂 common:
|  |  |  |- 📂 styles:
|  |  |  |  |- 📜 Home.module.css
|  |  |  |  |- 📜 globals.css
|  |  |  |- 📂 redux:
|  |  |  |  |- 📜 hooks.ts
|  |  |  |  |- 📜 train.ts
|  |  |  |  |- 📜 store.ts
|  |  |  |  |- 📜 backendApi.ts
|  |  |  |  |- 📜 userLogin.ts
|  |  |  |- 📂 components:
|  |  |  |  |- 📜 DlpTooltip.tsx
|  |  |  |  |- 📜 Footer.tsx
|  |  |  |  |- 📜 HtmlTooltip.tsx
|  |  |  |  |- 📜 NavBarMain.tsx
|  |  |  |  |- 📜 Spacer.tsx
|  |  |  |  |- 📜 EmailInput.tsx
|  |  |  |  |- 📜 TitleText.tsx
|  |  |  |  |- 📜 ClientOnlyPortal.tsx
|  |  |  |- 📂 utils:
|  |  |  |  |- 📜 dndHelpers.ts
|  |  |  |  |- 📜 firebase.ts
|  |  |  |  |- 📜 dateFormat.ts
|  |  |- 📂 backend_outputs:
|  |  |  |- 📜 model.pt : Last model.pt output
|  |  |  |- 📜 my_deep_learning_model.onnx : Last ONNX file output
|  |  |  |- 📜 model.pkl
|  |  |- 📜 next-env.d.ts
|  |  |- 📜 iris.csv : Sample CSV data
|  |  |- 📜 constants.ts
|  |  |- 📜 GlobalStyle.ts
|  |- 📂 public:
|  |  |- 📂 images:
|  |  |  |- 📂 wiki_images:
|  |  |  |  |- 📜 sigmoid_equation.png
|  |  |  |  |- 📜 batchnorm_diagram.png
|  |  |  |  |- 📜 maxpool2d.gif
|  |  |  |  |- 📜 conv2d.gif
|  |  |  |  |- 📜 conv2d2.gif
|  |  |  |  |- 📜 dropout_diagram.png
|  |  |  |  |- 📜 tanh_equation.png
|  |  |  |  |- 📜 avgpool_maxpool.gif
|  |  |  |  |- 📜 tanh_plot.png
|  |  |  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |  |- 📂 learn_mod_images:
|  |  |  |  |- 📜 neuralnet.png
|  |  |  |  |- 📜 robotImage.jpg
|  |  |  |  |- 📜 ReLUactivation.png
|  |  |  |  |- 📜 neuronWithEquation.png
|  |  |  |  |- 📜 tanhactivation.png
|  |  |  |  |- 📜 sigmoidactivation.png
|  |  |  |  |- 📜 binarystepactivation.png
|  |  |  |  |- 📜 lossExampleTable.png
|  |  |  |  |- 📜 sigmoidfunction.png
|  |  |  |  |- 📜 neuron.png
|  |  |  |  |- 📜 LeakyReLUactivation.png
|  |  |  |  |- 📜 lossExample.png
|  |  |  |  |- 📜 lossExampleEquation.png
|  |  |  |- 📂 logos:
|  |  |  |  |- 📂 dlp_branding:
|  |  |  |  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |- 📜 dsgt-logo-dark.png
|  |  |  |  |- 📜 google.png
|  |  |  |  |- 📜 react-logo.png
|  |  |  |  |- 📜 flask-logo.png
|  |  |  |  |- 📜 pytorch-logo.png
|  |  |  |  |- 📜 aws-logo.png
|  |  |  |  |- 📜 dsgt-logo-light.png
|  |  |  |  |- 📜 github.png
|  |  |  |  |- 📜 pandas-logo.png
|  |  |  |  |- 📜 python-logo.png
|  |  |  |  |- 📜 dsgt-logo-white-back.png
|  |  |  |- 📜 demo_video.gif : GIF tutorial of a simple classification training session
|  |  |- 📜 index.html : Base HTML file that will be initially rendered
|  |  |- 📜 robots.txt
|  |  |- 📜 manifest.json : Default React file for choosing icon based on
|  |  |- 📜 dlp-logo.ico : DLP Logo
|  |- 📜 package.json
|  |- 📜 .eslintrc.json
|  |- 📜 next-env.d.ts
|  |- 📜 tsconfig.json
|  |- 📜 yarn.lock
|  |- 📜 .eslintignore
|  |- 📜 next.config.js
|  |- 📜 jest.config.js
```

