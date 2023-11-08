# Architecture

## Backend Architecture

```
📦 backend
|  |- 📂 common:
|  |  |- 📜 optimizer.py : what optimizer to use (ie: SGD or Adam for now)
|  |  |- 📜 ai_drive.py
|  |  |- 📜 preprocessing.py
|  |  |- 📜 default_datasets.py : store logic to load in default datasets from scikit-learn
|  |  |- 📜 dataset.py : read in the dataset through URL or file upload
|  |  |- 📜 email_notifier.py : Endpoint to send email notification of training results via API Gateway + AWS SES
|  |  |- 📜 utils.py : utility functions that could be helpful
|  |  |- 📜 constants.py : list of helpful constants
|  |  |- 📜 __init__.py
|  |  |- 📜 kernel.py
|  |  |- 📜 loss_functions.py : loss function enum
|  |- 📂 ml:
|  |  |- 📜 ml_trainer.py : train a classical machine learning learning model on the dataset
|  |  |- 📜 __init__.py
|  |  |- 📜 ml_model_parser.py
|  |- 📂 dl:
|  |  |- 📜 dl_model.py : torch model based on user specifications from drag and drop
|  |  |- 📜 dl_eval.py : Evaluation functions for deep learning models in Pytorch (eg: accuracy, loss, etc)
|  |  |- 📜 dl_trainer.py : train a deep learning model on the dataset
|  |  |- 📜 detection.py
|  |  |- 📜 __init__.py
|  |  |- 📜 dl_model_parser.py : parse the user specified pytorch model
|  |- 📂 aws_helpers:
|  |  |- 📂 dynamo_db_utils:
|  |  |  |- 📜 trainspace_db.py
|  |  |  |- 📜 DynamoUnitTests.md
|  |  |  |- 📜 constants.py : list of helpful constants
|  |  |  |- 📜 dynamo_db_utils.py
|  |  |  |- 📜 userprogress_db.py
|  |  |- 📜 __init__.py
|  |- 📜 middleware.py
|  |- 📜 poetry.lock
|  |- 📜 epoch_times.csv
|  |- 📜 pyproject.toml
|  |- 📜 data.csv : data csv file for use in the playground
|  |- 📜 __init__.py
|  |- 📜 app.py : run the backend (entrypoint script)
```

## Frontend Architecture

```
📦 frontend
|  |- 📂 public:
|  |  |- 📂 images:
|  |  |  |- 📂 wiki_images:
|  |  |  |  |- 📜 tanh_plot.png
|  |  |  |  |- 📜 avgpool_maxpool.gif
|  |  |  |  |- 📜 conv2d2.gif
|  |  |  |  |- 📜 conv2d.gif
|  |  |  |  |- 📜 sigmoid_equation.png
|  |  |  |  |- 📜 maxpool2d.gif
|  |  |  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |  |  |- 📜 batchnorm_diagram.png
|  |  |  |  |- 📜 dropout_diagram.png
|  |  |  |  |- 📜 tanh_equation.png
|  |  |  |- 📂 learn_mod_images:
|  |  |  |  |- 📜 neuron.png
|  |  |  |  |- 📜 tanhactivation.png
|  |  |  |  |- 📜 neuronWithEquation.png
|  |  |  |  |- 📜 sigmoidactivation.png
|  |  |  |  |- 📜 lossExampleTable.png
|  |  |  |  |- 📜 sigmoidfunction.png
|  |  |  |  |- 📜 lossExampleEquation.png
|  |  |  |  |- 📜 neuralnet.png
|  |  |  |  |- 📜 LeakyReLUactivation.png
|  |  |  |  |- 📜 ReLUactivation.png
|  |  |  |  |- 📜 robotImage.jpg
|  |  |  |  |- 📜 binarystepactivation.png
|  |  |  |  |- 📜 lossExample.png
|  |  |  |- 📂 logos:
|  |  |  |  |- 📂 dlp_branding:
|  |  |  |  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |- 📜 react-logo.png
|  |  |  |  |- 📜 google.png
|  |  |  |  |- 📜 pytorch-logo.png
|  |  |  |  |- 📜 pandas-logo.png
|  |  |  |  |- 📜 dsgt-logo-dark.png
|  |  |  |  |- 📜 dsgt-logo-light.png
|  |  |  |  |- 📜 python-logo.png
|  |  |  |  |- 📜 dsgt-logo-white-back.png
|  |  |  |  |- 📜 github.png
|  |  |  |  |- 📜 aws-logo.png
|  |  |  |  |- 📜 flask-logo.png
|  |  |  |- 📜 demo_video.gif : GIF tutorial of a simple classification training session
|  |  |- 📜 robots.txt
|  |  |- 📜 dlp-logo.ico : DLP Logo
|  |  |- 📜 index.html : Base HTML file that will be initially rendered
|  |  |- 📜 manifest.json : Default React file for choosing icon based on
|  |- 📂 layer_docs:
|  |  |- 📜 Linear.md : Doc for Linear layer
|  |  |- 📜 ReLU.md : Doc for ReLU later
|  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |- 📜 Softmax.md : Doc for Softmax layer
|  |- 📂 src:
|  |  |- 📂 pages:
|  |  |  |- 📂 train:
|  |  |  |  |- 📜 index.tsx
|  |  |  |  |- 📜 [train_space_id].tsx
|  |  |  |- 📜 _app.tsx
|  |  |  |- 📜 _document.tsx
|  |  |  |- 📜 LearnContent.tsx
|  |  |  |- 📜 wiki.tsx
|  |  |  |- 📜 learn.tsx
|  |  |  |- 📜 forgot.tsx
|  |  |  |- 📜 settings.tsx
|  |  |  |- 📜 dashboard.tsx
|  |  |  |- 📜 login.tsx
|  |  |  |- 📜 about.tsx
|  |  |  |- 📜 feedback.tsx
|  |  |- 📂 common:
|  |  |  |- 📂 styles:
|  |  |  |  |- 📜 globals.css
|  |  |  |  |- 📜 Home.module.css
|  |  |  |- 📂 redux:
|  |  |  |  |- 📜 userLogin.ts
|  |  |  |  |- 📜 backendApi.ts
|  |  |  |  |- 📜 store.ts
|  |  |  |  |- 📜 hooks.ts
|  |  |  |  |- 📜 train.ts
|  |  |  |- 📂 utils:
|  |  |  |  |- 📜 dndHelpers.ts
|  |  |  |  |- 📜 firebase.ts
|  |  |  |  |- 📜 dateFormat.ts
|  |  |  |- 📂 components:
|  |  |  |  |- 📜 Spacer.tsx
|  |  |  |  |- 📜 Footer.tsx
|  |  |  |  |- 📜 HtmlTooltip.tsx
|  |  |  |  |- 📜 DlpTooltip.tsx
|  |  |  |  |- 📜 ClientOnlyPortal.tsx
|  |  |  |  |- 📜 NavBarMain.tsx
|  |  |  |  |- 📜 EmailInput.tsx
|  |  |  |  |- 📜 TitleText.tsx
|  |  |- 📂 backend_outputs:
|  |  |  |- 📜 my_deep_learning_model.onnx : Last ONNX file output
|  |  |  |- 📜 model.pt : Last model.pt output
|  |  |  |- 📜 model.pkl
|  |  |- 📂 features:
|  |  |  |- 📂 Train:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 trainspaceSlice.ts
|  |  |  |  |  |- 📜 trainspaceApi.ts
|  |  |  |  |- 📂 features:
|  |  |  |  |  |- 📂 Image:
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 imageActions.ts
|  |  |  |  |  |  |  |- 📜 imageApi.ts
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 imageTypes.ts
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 imageConstants.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 ImageParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 ImageFlow.tsx
|  |  |  |  |  |  |  |- 📜 ImageDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageReviewStep.tsx
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |  |- 📂 Tabular:
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 tabularActions.ts
|  |  |  |  |  |  |  |- 📜 tabularApi.ts
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 tabularTypes.ts
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 tabularConstants.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 TabularParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 TabularDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularFlow.tsx
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |- 📂 types:
|  |  |  |  |  |- 📜 trainTypes.ts
|  |  |  |  |- 📂 constants:
|  |  |  |  |  |- 📜 trainConstants.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 CreateTrainspace.tsx
|  |  |  |  |  |- 📜 TrainspaceLayout.tsx
|  |  |  |  |  |- 📜 DatasetStepLayout.tsx
|  |  |  |- 📂 OpenAi:
|  |  |  |  |- 📜 openAiUtils.ts
|  |  |  |- 📂 Dashboard:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 dashboardApi.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TrainBarChart.tsx
|  |  |  |  |  |- 📜 TrainDataGrid.tsx
|  |  |  |  |  |- 📜 TrainDoughnutChart.tsx
|  |  |  |- 📂 Feedback:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 feedbackApi.ts
|  |  |  |- 📂 LearnMod:
|  |  |  |  |- 📜 FRQuestion.tsx
|  |  |  |  |- 📜 LearningModulesContent.tsx
|  |  |  |  |- 📜 ModulesSideBar.tsx
|  |  |  |  |- 📜 MCQuestion.tsx
|  |  |  |  |- 📜 ClassCard.tsx
|  |  |  |  |- 📜 ImageComponent.tsx
|  |  |  |  |- 📜 Exercise.tsx
|  |  |- 📜 next-env.d.ts
|  |  |- 📜 iris.csv : Sample CSV data
|  |  |- 📜 GlobalStyle.ts
|  |  |- 📜 constants.ts
|  |- 📜 next-env.d.ts
|  |- 📜 tsconfig.json
|  |- 📜 next.config.js
|  |- 📜 jest.config.js
|  |- 📜 .eslintrc.json
|  |- 📜 pnpm-lock.yaml
|  |- 📜 package.json
|  |- 📜 .eslintignore
```

