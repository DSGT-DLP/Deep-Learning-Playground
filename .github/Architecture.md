# Architecture

## Backend Architecture

```
📦 backend
|  |- 📂 aws_helpers:
|  |  |- 📂 dynamo_db_utils:
|  |  |  |- 📜 userprogress_db.py
|  |  |  |- 📜 DynamoUnitTests.md
|  |  |  |- 📜 dynamo_db_utils.py
|  |  |  |- 📜 constants.py : list of helpful constants
|  |  |  |- 📜 trainspace_db.py
|  |  |- 📜 __init__.py
|  |- 📂 dl:
|  |  |- 📜 dl_eval.py : Evaluation functions for deep learning models in Pytorch (eg: accuracy, loss, etc)
|  |  |- 📜 dl_model_parser.py : parse the user specified pytorch model
|  |  |- 📜 dl_trainer.py : train a deep learning model on the dataset
|  |  |- 📜 detection.py
|  |  |- 📜 dl_model.py : torch model based on user specifications from drag and drop
|  |  |- 📜 __init__.py
|  |- 📂 common:
|  |  |- 📜 dataset.py : read in the dataset through URL or file upload
|  |  |- 📜 default_datasets.py : store logic to load in default datasets from scikit-learn
|  |  |- 📜 preprocessing.py
|  |  |- 📜 kernel.py
|  |  |- 📜 loss_functions.py : loss function enum
|  |  |- 📜 email_notifier.py : Endpoint to send email notification of training results via API Gateway + AWS SES
|  |  |- 📜 constants.py : list of helpful constants
|  |  |- 📜 utils.py : utility functions that could be helpful
|  |  |- 📜 optimizer.py : what optimizer to use (ie: SGD or Adam for now)
|  |  |- 📜 __init__.py
|  |  |- 📜 ai_drive.py
|  |- 📂 ml:
|  |  |- 📜 ml_model_parser.py
|  |  |- 📜 ml_trainer.py : train a classical machine learning learning model on the dataset
|  |  |- 📜 __init__.py
|  |- 📜 pyproject.toml
|  |- 📜 epoch_times.csv
|  |- 📜 data.csv : data csv file for use in the playground
|  |- 📜 poetry.lock
|  |- 📜 middleware.py
|  |- 📜 app.py : run the backend (entrypoint script)
```

## Frontend Architecture

```
📦 frontend
|  |- 📂 src:
|  |  |- 📂 pages:
|  |  |  |- 📂 train:
|  |  |  |  |- 📜 index.tsx
|  |  |  |  |- 📜 [train_space_id].tsx
|  |  |  |- 📜 LearnContent.tsx
|  |  |  |- 📜 feedback.tsx
|  |  |  |- 📜 _document.tsx
|  |  |  |- 📜 learn.tsx
|  |  |  |- 📜 login.tsx
|  |  |  |- 📜 about.tsx
|  |  |  |- 📜 _app.tsx
|  |  |  |- 📜 settings.tsx
|  |  |  |- 📜 forgot.tsx
|  |  |  |- 📜 wiki.tsx
|  |  |  |- 📜 dashboard.tsx
|  |  |- 📂 backend_outputs:
|  |  |  |- 📜 model.pkl
|  |  |  |- 📜 model.pt : Last model.pt output
|  |  |  |- 📜 my_deep_learning_model.onnx : Last ONNX file output
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
|  |  |  |- 📂 LearnMod:
|  |  |  |  |- 📜 ImageComponent.tsx
|  |  |  |  |- 📜 ModulesSideBar.tsx
|  |  |  |  |- 📜 ClassCard.tsx
|  |  |  |  |- 📜 LearningModulesContent.tsx
|  |  |  |  |- 📜 MCQuestion.tsx
|  |  |  |  |- 📜 FRQuestion.tsx
|  |  |  |  |- 📜 Exercise.tsx
|  |  |  |- 📂 OpenAi:
|  |  |  |  |- 📜 openAiUtils.ts
|  |  |  |- 📂 Train:
|  |  |  |  |- 📂 types:
|  |  |  |  |  |- 📜 trainTypes.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 CreateTrainspace.tsx
|  |  |  |  |  |- 📜 TrainspaceLayout.tsx
|  |  |  |  |  |- 📜 DatasetStepLayout.tsx
|  |  |  |  |- 📂 features:
|  |  |  |  |  |- 📂 Tabular:
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 tabularTypes.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 TabularReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 TabularParametersStep.tsx
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 tabularConstants.ts
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 tabularActions.ts
|  |  |  |  |  |  |  |- 📜 tabularApi.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |  |- 📂 Image:
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 imageTypes.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 ImageTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 ImageParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageReviewStep.tsx
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 imageConstants.ts
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 imageActions.ts
|  |  |  |  |  |  |  |- 📜 imageApi.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |- 📂 constants:
|  |  |  |  |  |- 📜 trainConstants.ts
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 trainspaceSlice.ts
|  |  |  |  |  |- 📜 trainspaceApi.ts
|  |  |- 📂 common:
|  |  |  |- 📂 components:
|  |  |  |  |- 📜 EmailInput.tsx
|  |  |  |  |- 📜 ClientOnlyPortal.tsx
|  |  |  |  |- 📜 Footer.tsx
|  |  |  |  |- 📜 Spacer.tsx
|  |  |  |  |- 📜 TitleText.tsx
|  |  |  |  |- 📜 NavBarMain.tsx
|  |  |  |- 📂 utils:
|  |  |  |  |- 📜 dateFormat.ts
|  |  |  |  |- 📜 firebase.ts
|  |  |  |  |- 📜 dndHelpers.ts
|  |  |  |- 📂 redux:
|  |  |  |  |- 📜 userLogin.ts
|  |  |  |  |- 📜 train.ts
|  |  |  |  |- 📜 backendApi.ts
|  |  |  |  |- 📜 store.ts
|  |  |  |  |- 📜 hooks.ts
|  |  |  |- 📂 styles:
|  |  |  |  |- 📜 globals.css
|  |  |  |  |- 📜 Home.module.css
|  |  |- 📜 GlobalStyle.ts
|  |  |- 📜 next-env.d.ts
|  |  |- 📜 iris.csv : Sample CSV data
|  |  |- 📜 constants.ts
|  |- 📂 layer_docs:
|  |  |- 📜 Linear.md : Doc for Linear layer
|  |  |- 📜 Softmax.md : Doc for Softmax layer
|  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |- 📜 ReLU.md : Doc for ReLU later
|  |- 📂 public:
|  |  |- 📂 images:
|  |  |  |- 📂 wiki_images:
|  |  |  |  |- 📜 tanh_plot.png
|  |  |  |  |- 📜 sigmoid_equation.png
|  |  |  |  |- 📜 avgpool_maxpool.gif
|  |  |  |  |- 📜 maxpool2d.gif
|  |  |  |  |- 📜 batchnorm_diagram.png
|  |  |  |  |- 📜 conv2d2.gif
|  |  |  |  |- 📜 conv2d.gif
|  |  |  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |  |  |- 📜 dropout_diagram.png
|  |  |  |  |- 📜 tanh_equation.png
|  |  |  |- 📂 logos:
|  |  |  |  |- 📂 dlp_branding:
|  |  |  |  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |- 📜 google.png
|  |  |  |  |- 📜 dsgt-logo-white-back.png
|  |  |  |  |- 📜 dsgt-logo-dark.png
|  |  |  |  |- 📜 dsgt-logo-light.png
|  |  |  |  |- 📜 pytorch-logo.png
|  |  |  |  |- 📜 github.png
|  |  |  |  |- 📜 aws-logo.png
|  |  |  |  |- 📜 python-logo.png
|  |  |  |  |- 📜 pandas-logo.png
|  |  |  |  |- 📜 react-logo.png
|  |  |  |  |- 📜 flask-logo.png
|  |  |  |- 📂 learn_mod_images:
|  |  |  |  |- 📜 robotImage.jpg
|  |  |  |  |- 📜 sigmoidfunction.png
|  |  |  |  |- 📜 lossExampleEquation.png
|  |  |  |  |- 📜 LeakyReLUactivation.png
|  |  |  |  |- 📜 lossExample.png
|  |  |  |  |- 📜 neuron.png
|  |  |  |  |- 📜 tanhactivation.png
|  |  |  |  |- 📜 neuronWithEquation.png
|  |  |  |  |- 📜 lossExampleTable.png
|  |  |  |  |- 📜 ReLUactivation.png
|  |  |  |  |- 📜 neuralnet.png
|  |  |  |  |- 📜 sigmoidactivation.png
|  |  |  |  |- 📜 binarystepactivation.png
|  |  |  |- 📜 demo_video.gif : GIF tutorial of a simple classification training session
|  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |- 📜 robots.txt
|  |  |- 📜 index.html : Base HTML file that will be initially rendered
|  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |- 📜 dlp-logo.ico : DLP Logo
|  |  |- 📜 manifest.json : Default React file for choosing icon based on
|  |- 📜 .eslintrc.json
|  |- 📜 package.json
|  |- 📜 babel.config.js
|  |- 📜 next.config.js
|  |- 📜 jest.config.js
|  |- 📜 yarn.lock
|  |- 📜 .eslintignore
|  |- 📜 next-env.d.ts
|  |- 📜 tsconfig.json
```

