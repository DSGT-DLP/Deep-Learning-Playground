# Architecture

## Backend Architecture

```
📦 backend
|  |- 📂 ml:
|  |  |- 📜 __init__.py
|  |  |- 📜 ml_trainer.py : train a classical machine learning learning model on the dataset
|  |  |- 📜 ml_model_parser.py
|  |- 📂 dl:
|  |  |- 📜 dl_trainer.py : train a deep learning model on the dataset
|  |  |- 📜 __init__.py
|  |  |- 📜 dl_model.py : torch model based on user specifications from drag and drop
|  |  |- 📜 detection.py
|  |  |- 📜 dl_model_parser.py : parse the user specified pytorch model
|  |  |- 📜 dl_eval.py : Evaluation functions for deep learning models in Pytorch (eg: accuracy, loss, etc)
|  |- 📂 common:
|  |  |- 📜 dataset.py : read in the dataset through URL or file upload
|  |  |- 📜 optimizer.py : what optimizer to use (ie: SGD or Adam for now)
|  |  |- 📜 loss_functions.py : loss function enum
|  |  |- 📜 kernel.py
|  |  |- 📜 __init__.py
|  |  |- 📜 ai_drive.py
|  |  |- 📜 default_datasets.py : store logic to load in default datasets from scikit-learn
|  |  |- 📜 utils.py : utility functions that could be helpful
|  |  |- 📜 constants.py : list of helpful constants
|  |  |- 📜 email_notifier.py : Endpoint to send email notification of training results via API Gateway + AWS SES
|  |  |- 📜 preprocessing.py
|  |- 📂 aws_helpers:
|  |  |- 📂 dynamo_db_utils:
|  |  |  |- 📜 trainspace_db.py
|  |  |  |- 📜 constants.py : list of helpful constants
|  |  |  |- 📜 dynamo_db_utils.py
|  |  |  |- 📜 userprogress_db.py
|  |  |  |- 📜 DynamoUnitTests.md
|  |  |- 📜 __init__.py
|  |- 📜 poetry.lock
|  |- 📜 __init__.py
|  |- 📜 middleware.py
|  |- 📜 pyproject.toml
|  |- 📜 epoch_times.csv
|  |- 📜 app.py : run the backend (entrypoint script)
|  |- 📜 data.csv : data csv file for use in the playground
```

## Frontend Architecture

```
📦 frontend
|  |- 📂 src:
|  |  |- 📂 pages:
|  |  |  |- 📂 train:
|  |  |  |  |- 📜 [train_space_id].tsx
|  |  |  |  |- 📜 index.tsx
|  |  |  |- 📜 login.tsx
|  |  |  |- 📜 learn.tsx
|  |  |  |- 📜 about.tsx
|  |  |  |- 📜 _document.tsx
|  |  |  |- 📜 feedback.tsx
|  |  |  |- 📜 dashboard.tsx
|  |  |  |- 📜 LearnContent.tsx
|  |  |  |- 📜 wiki.tsx
|  |  |  |- 📜 _app.tsx
|  |  |  |- 📜 forgot.tsx
|  |  |  |- 📜 settings.tsx
|  |  |- 📂 common:
|  |  |  |- 📂 utils:
|  |  |  |  |- 📜 dateFormat.ts
|  |  |  |  |- 📜 firebase.ts
|  |  |  |  |- 📜 dndHelpers.ts
|  |  |  |- 📂 redux:
|  |  |  |  |- 📜 store.ts
|  |  |  |  |- 📜 train.ts
|  |  |  |  |- 📜 userLogin.ts
|  |  |  |  |- 📜 backendApi.ts
|  |  |  |  |- 📜 hooks.ts
|  |  |  |- 📂 styles:
|  |  |  |  |- 📜 globals.css
|  |  |  |  |- 📜 Home.module.css
|  |  |  |- 📂 components:
|  |  |  |  |- 📜 Spacer.tsx
|  |  |  |  |- 📜 Footer.tsx
|  |  |  |  |- 📜 NavBarMain.tsx
|  |  |  |  |- 📜 HtmlTooltip.tsx
|  |  |  |  |- 📜 ClientOnlyPortal.tsx
|  |  |  |  |- 📜 EmailInput.tsx
|  |  |  |  |- 📜 DlpTooltip.tsx
|  |  |  |  |- 📜 TitleText.tsx
|  |  |- 📂 backend_outputs:
|  |  |  |- 📜 my_deep_learning_model.onnx : Last ONNX file output
|  |  |  |- 📜 model.pkl
|  |  |  |- 📜 model.pt : Last model.pt output
|  |  |- 📂 features:
|  |  |  |- 📂 Train:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 trainspaceSlice.ts
|  |  |  |  |  |- 📜 trainspaceApi.ts
|  |  |  |  |- 📂 constants:
|  |  |  |  |  |- 📜 trainConstants.ts
|  |  |  |  |- 📂 types:
|  |  |  |  |  |- 📜 trainTypes.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TrainspaceLayout.tsx
|  |  |  |  |  |- 📜 CreateTrainspace.tsx
|  |  |  |  |  |- 📜 DatasetStepLayout.tsx
|  |  |  |  |- 📂 features:
|  |  |  |  |  |- 📂 Image:
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 imageActions.ts
|  |  |  |  |  |  |  |- 📜 imageApi.ts
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 imageConstants.ts
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 imageTypes.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 ImageReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageFlow.tsx
|  |  |  |  |  |  |  |- 📜 ImageDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageTrainspace.tsx
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |  |- 📂 Tabular:
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 tabularActions.ts
|  |  |  |  |  |  |  |- 📜 tabularApi.ts
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 tabularConstants.ts
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 tabularTypes.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 TabularTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 TabularReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularFlow.tsx
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |- 📂 Feedback:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 feedbackApi.ts
|  |  |  |- 📂 LearnMod:
|  |  |  |  |- 📜 MCQuestion.tsx
|  |  |  |  |- 📜 Exercise.tsx
|  |  |  |  |- 📜 ModulesSideBar.tsx
|  |  |  |  |- 📜 FRQuestion.tsx
|  |  |  |  |- 📜 ClassCard.tsx
|  |  |  |  |- 📜 ImageComponent.tsx
|  |  |  |  |- 📜 LearningModulesContent.tsx
|  |  |  |- 📂 OpenAi:
|  |  |  |  |- 📜 openAiUtils.ts
|  |  |  |- 📂 Dashboard:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 dashboardApi.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TrainDoughnutChart.tsx
|  |  |  |  |  |- 📜 TrainBarChart.tsx
|  |  |  |  |  |- 📜 TrainDataGrid.tsx
|  |  |- 📜 constants.ts
|  |  |- 📜 GlobalStyle.ts
|  |  |- 📜 next-env.d.ts
|  |  |- 📜 iris.csv : Sample CSV data
|  |- 📂 public:
|  |  |- 📂 images:
|  |  |  |- 📂 learn_mod_images:
|  |  |  |  |- 📜 binarystepactivation.png
|  |  |  |  |- 📜 neuron.png
|  |  |  |  |- 📜 lossExampleEquation.png
|  |  |  |  |- 📜 ReLUactivation.png
|  |  |  |  |- 📜 neuronWithEquation.png
|  |  |  |  |- 📜 lossExampleTable.png
|  |  |  |  |- 📜 robotImage.jpg
|  |  |  |  |- 📜 lossExample.png
|  |  |  |  |- 📜 LeakyReLUactivation.png
|  |  |  |  |- 📜 tanhactivation.png
|  |  |  |  |- 📜 sigmoidactivation.png
|  |  |  |  |- 📜 neuralnet.png
|  |  |  |  |- 📜 sigmoidfunction.png
|  |  |  |- 📂 wiki_images:
|  |  |  |  |- 📜 tanh_equation.png
|  |  |  |  |- 📜 tanh_plot.png
|  |  |  |  |- 📜 conv2d.gif
|  |  |  |  |- 📜 conv2d2.gif
|  |  |  |  |- 📜 dropout_diagram.png
|  |  |  |  |- 📜 batchnorm_diagram.png
|  |  |  |  |- 📜 maxpool2d.gif
|  |  |  |  |- 📜 avgpool_maxpool.gif
|  |  |  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |  |  |- 📜 sigmoid_equation.png
|  |  |  |- 📂 logos:
|  |  |  |  |- 📂 dlp_branding:
|  |  |  |  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |- 📜 pytorch-logo.png
|  |  |  |  |- 📜 dsgt-logo-white-back.png
|  |  |  |  |- 📜 flask-logo.png
|  |  |  |  |- 📜 aws-logo.png
|  |  |  |  |- 📜 google.png
|  |  |  |  |- 📜 python-logo.png
|  |  |  |  |- 📜 dsgt-logo-dark.png
|  |  |  |  |- 📜 pandas-logo.png
|  |  |  |  |- 📜 react-logo.png
|  |  |  |  |- 📜 github.png
|  |  |  |  |- 📜 dsgt-logo-light.png
|  |  |  |- 📜 demo_video.gif : GIF tutorial of a simple classification training session
|  |  |- 📜 manifest.json : Default React file for choosing icon based on
|  |  |- 📜 index.html : Base HTML file that will be initially rendered
|  |  |- 📜 robots.txt
|  |  |- 📜 dlp-logo.ico : DLP Logo
|  |- 📂 layer_docs:
|  |  |- 📜 Linear.md : Doc for Linear layer
|  |  |- 📜 Softmax.md : Doc for Softmax layer
|  |  |- 📜 ReLU.md : Doc for ReLU later
|  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |- 📜 tsconfig.json
|  |- 📜 package.json
|  |- 📜 jest.config.js
|  |- 📜 .eslintrc.json
|  |- 📜 next.config.js
|  |- 📜 .eslintignore
|  |- 📜 pnpm-lock.yaml
|  |- 📜 next-env.d.ts
```

