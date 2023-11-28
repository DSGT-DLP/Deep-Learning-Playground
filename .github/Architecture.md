# Architecture

## Backend Architecture

```
📦 backend
|  |- 📂 common:
|  |  |- 📜 default_datasets.py : store logic to load in default datasets from scikit-learn
|  |  |- 📜 ai_drive.py
|  |  |- 📜 optimizer.py : what optimizer to use (ie: SGD or Adam for now)
|  |  |- 📜 __init__.py
|  |  |- 📜 kernel.py
|  |  |- 📜 constants.py : list of helpful constants
|  |  |- 📜 dataset.py : read in the dataset through URL or file upload
|  |  |- 📜 preprocessing.py
|  |  |- 📜 utils.py : utility functions that could be helpful
|  |  |- 📜 email_notifier.py : Endpoint to send email notification of training results via API Gateway + AWS SES
|  |  |- 📜 loss_functions.py : loss function enum
|  |- 📂 aws_helpers:
|  |  |- 📂 dynamo_db_utils:
|  |  |  |- 📜 userprogress_db.py
|  |  |  |- 📜 constants.py : list of helpful constants
|  |  |  |- 📜 DynamoUnitTests.md
|  |  |  |- 📜 trainspace_db.py
|  |  |  |- 📜 dynamo_db_utils.py
|  |  |- 📜 __init__.py
|  |- 📂 ml:
|  |  |- 📜 ml_trainer.py : train a classical machine learning learning model on the dataset
|  |  |- 📜 ml_model_parser.py
|  |  |- 📜 __init__.py
|  |- 📂 dl:
|  |  |- 📜 dl_model_parser.py : parse the user specified pytorch model
|  |  |- 📜 dl_model.py : torch model based on user specifications from drag and drop
|  |  |- 📜 __init__.py
|  |  |- 📜 dl_eval.py : Evaluation functions for deep learning models in Pytorch (eg: accuracy, loss, etc)
|  |  |- 📜 dl_trainer.py : train a deep learning model on the dataset
|  |  |- 📜 detection.py
|  |- 📜 __init__.py
|  |- 📜 data.csv : data csv file for use in the playground
|  |- 📜 pyproject.toml
|  |- 📜 middleware.py
|  |- 📜 poetry.lock
|  |- 📜 app.py : run the backend (entrypoint script)
|  |- 📜 epoch_times.csv
```

## Frontend Architecture

```
📦 frontend
|  |- 📂 layer_docs:
|  |  |- 📜 Softmax.md : Doc for Softmax layer
|  |  |- 📜 Linear.md : Doc for Linear layer
|  |  |- 📜 ReLU.md : Doc for ReLU later
|  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |- 📂 public:
|  |  |- 📂 images:
|  |  |  |- 📂 wiki_images:
|  |  |  |  |- 📜 tanh_plot.png
|  |  |  |  |- 📜 dropout_diagram.png
|  |  |  |  |- 📜 tanh_equation.png
|  |  |  |  |- 📜 conv2d2.gif
|  |  |  |  |- 📜 batchnorm_diagram.png
|  |  |  |  |- 📜 conv2d.gif
|  |  |  |  |- 📜 maxpool2d.gif
|  |  |  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |  |  |- 📜 sigmoid_equation.png
|  |  |  |  |- 📜 avgpool_maxpool.gif
|  |  |  |- 📂 learn_mod_images:
|  |  |  |  |- 📜 lossExample.png
|  |  |  |  |- 📜 robotImage.jpg
|  |  |  |  |- 📜 LeakyReLUactivation.png
|  |  |  |  |- 📜 neuralnet.png
|  |  |  |  |- 📜 lossExampleTable.png
|  |  |  |  |- 📜 binarystepactivation.png
|  |  |  |  |- 📜 ReLUactivation.png
|  |  |  |  |- 📜 sigmoidfunction.png
|  |  |  |  |- 📜 tanhactivation.png
|  |  |  |  |- 📜 neuronWithEquation.png
|  |  |  |  |- 📜 neuron.png
|  |  |  |  |- 📜 lossExampleEquation.png
|  |  |  |  |- 📜 sigmoidactivation.png
|  |  |  |- 📂 logos:
|  |  |  |  |- 📂 dlp_branding:
|  |  |  |  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |- 📜 dsgt-logo-light.png
|  |  |  |  |- 📜 pandas-logo.png
|  |  |  |  |- 📜 github.png
|  |  |  |  |- 📜 flask-logo.png
|  |  |  |  |- 📜 react-logo.png
|  |  |  |  |- 📜 aws-logo.png
|  |  |  |  |- 📜 pytorch-logo.png
|  |  |  |  |- 📜 dsgt-logo-white-back.png
|  |  |  |  |- 📜 dsgt-logo-dark.png
|  |  |  |  |- 📜 python-logo.png
|  |  |  |  |- 📜 google.png
|  |  |  |- 📜 demo_video.gif : GIF tutorial of a simple classification training session
|  |  |- 📜 manifest.json : Default React file for choosing icon based on
|  |  |- 📜 index.html : Base HTML file that will be initially rendered
|  |  |- 📜 dlp-logo.ico : DLP Logo
|  |  |- 📜 robots.txt
|  |- 📂 src:
|  |  |- 📂 backend_outputs:
|  |  |  |- 📜 model.pt : Last model.pt output
|  |  |  |- 📜 my_deep_learning_model.onnx : Last ONNX file output
|  |  |  |- 📜 model.pkl
|  |  |- 📂 common:
|  |  |  |- 📂 redux:
|  |  |  |  |- 📜 userLogin.ts
|  |  |  |  |- 📜 train.ts
|  |  |  |  |- 📜 backendApi.ts
|  |  |  |  |- 📜 store.ts
|  |  |  |  |- 📜 hooks.ts
|  |  |  |- 📂 styles:
|  |  |  |  |- 📜 Home.module.css
|  |  |  |  |- 📜 globals.css
|  |  |  |- 📂 components:
|  |  |  |  |- 📜 Spacer.tsx
|  |  |  |  |- 📜 Footer.tsx
|  |  |  |  |- 📜 TitleText.tsx
|  |  |  |  |- 📜 DlpTooltip.tsx
|  |  |  |  |- 📜 HtmlTooltip.tsx
|  |  |  |  |- 📜 EmailInput.tsx
|  |  |  |  |- 📜 NavBarMain.tsx
|  |  |  |  |- 📜 ClientOnlyPortal.tsx
|  |  |  |- 📂 utils:
|  |  |  |  |- 📜 firebase.ts
|  |  |  |  |- 📜 dndHelpers.ts
|  |  |  |  |- 📜 dateFormat.ts
|  |  |- 📂 features:
|  |  |  |- 📂 Train:
|  |  |  |  |- 📂 constants:
|  |  |  |  |  |- 📜 trainConstants.ts
|  |  |  |  |- 📂 types:
|  |  |  |  |  |- 📜 trainTypes.ts
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 trainspaceSlice.ts
|  |  |  |  |  |- 📜 trainspaceApi.ts
|  |  |  |  |- 📂 features:
|  |  |  |  |  |- 📂 Image:
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 imageConstants.ts
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 imageTypes.ts
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 imageApi.ts
|  |  |  |  |  |  |  |- 📜 imageActions.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 ImageFlow.tsx
|  |  |  |  |  |  |  |- 📜 ImageDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 ImageReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageParametersStep.tsx
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |  |- 📂 Tabular:
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 tabularConstants.ts
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 tabularTypes.ts
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 tabularActions.ts
|  |  |  |  |  |  |  |- 📜 tabularApi.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 TabularParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 TabularFlow.tsx
|  |  |  |  |  |  |  |- 📜 TabularReviewStep.tsx
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 CreateTrainspace.tsx
|  |  |  |  |  |- 📜 DatasetStepLayout.tsx
|  |  |  |  |  |- 📜 TrainspaceLayout.tsx
|  |  |  |- 📂 Feedback:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 feedbackApi.ts
|  |  |  |- 📂 Dashboard:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 dashboardApi.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TrainDataGrid.tsx
|  |  |  |  |  |- 📜 TrainBarChart.tsx
|  |  |  |  |  |- 📜 TrainDoughnutChart.tsx
|  |  |  |- 📂 LearnMod:
|  |  |  |  |- 📜 MCQuestion.tsx
|  |  |  |  |- 📜 ImageComponent.tsx
|  |  |  |  |- 📜 ModulesSideBar.tsx
|  |  |  |  |- 📜 Exercise.tsx
|  |  |  |  |- 📜 FRQuestion.tsx
|  |  |  |  |- 📜 LearningModulesContent.tsx
|  |  |  |  |- 📜 ClassCard.tsx
|  |  |  |- 📂 OpenAi:
|  |  |  |  |- 📜 openAiUtils.ts
|  |  |- 📂 pages:
|  |  |  |- 📂 train:
|  |  |  |  |- 📜 index.tsx
|  |  |  |  |- 📜 [train_space_id].tsx
|  |  |  |- 📜 login.tsx
|  |  |  |- 📜 feedback.tsx
|  |  |  |- 📜 learn.tsx
|  |  |  |- 📜 dashboard.tsx
|  |  |  |- 📜 about.tsx
|  |  |  |- 📜 _document.tsx
|  |  |  |- 📜 _app.tsx
|  |  |  |- 📜 settings.tsx
|  |  |  |- 📜 forgot.tsx
|  |  |  |- 📜 LearnContent.tsx
|  |  |  |- 📜 wiki.tsx
|  |  |- 📜 iris.csv : Sample CSV data
|  |  |- 📜 GlobalStyle.ts
|  |  |- 📜 constants.ts
|  |  |- 📜 next-env.d.ts
|  |- 📜 next.config.js
|  |- 📜 jest.config.js
|  |- 📜 .eslintrc.json
|  |- 📜 package.json
|  |- 📜 tsconfig.json
|  |- 📜 pnpm-lock.yaml
|  |- 📜 next-env.d.ts
|  |- 📜 .eslintignore
```

