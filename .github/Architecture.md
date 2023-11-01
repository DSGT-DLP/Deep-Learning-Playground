# Architecture

## Backend Architecture

```
📦 backend
|  |- 📂 dl:
|  |  |- 📜 detection.py
|  |  |- 📜 dl_trainer.py : train a deep learning model on the dataset
|  |  |- 📜 dl_model_parser.py : parse the user specified pytorch model
|  |  |- 📜 __init__.py
|  |  |- 📜 dl_eval.py : Evaluation functions for deep learning models in Pytorch (eg: accuracy, loss, etc)
|  |  |- 📜 dl_model.py : torch model based on user specifications from drag and drop
|  |- 📂 aws_helpers:
|  |  |- 📂 dynamo_db_utils:
|  |  |  |- 📜 DynamoUnitTests.md
|  |  |  |- 📜 constants.py : list of helpful constants
|  |  |  |- 📜 userprogress_db.py
|  |  |  |- 📜 trainspace_db.py
|  |  |  |- 📜 dynamo_db_utils.py
|  |  |- 📜 __init__.py
|  |- 📂 ml:
|  |  |- 📜 __init__.py
|  |  |- 📜 ml_model_parser.py
|  |  |- 📜 ml_trainer.py : train a classical machine learning learning model on the dataset
|  |- 📂 common:
|  |  |- 📜 loss_functions.py : loss function enum
|  |  |- 📜 preprocessing.py
|  |  |- 📜 constants.py : list of helpful constants
|  |  |- 📜 dataset.py : read in the dataset through URL or file upload
|  |  |- 📜 optimizer.py : what optimizer to use (ie: SGD or Adam for now)
|  |  |- 📜 __init__.py
|  |  |- 📜 email_notifier.py : Endpoint to send email notification of training results via API Gateway + AWS SES
|  |  |- 📜 ai_drive.py
|  |  |- 📜 default_datasets.py : store logic to load in default datasets from scikit-learn
|  |  |- 📜 kernel.py
|  |  |- 📜 utils.py : utility functions that could be helpful
|  |- 📜 app.py : run the backend (entrypoint script)
|  |- 📜 __init__.py
|  |- 📜 data.csv : data csv file for use in the playground
|  |- 📜 pyproject.toml
|  |- 📜 epoch_times.csv
|  |- 📜 middleware.py
|  |- 📜 poetry.lock
```

## Frontend Architecture

```
📦 frontend
|  |- 📂 layer_docs:
|  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |- 📜 ReLU.md : Doc for ReLU later
|  |  |- 📜 Softmax.md : Doc for Softmax layer
|  |  |- 📜 Linear.md : Doc for Linear layer
|  |- 📂 src:
|  |  |- 📂 features:
|  |  |  |- 📂 Train:
|  |  |  |  |- 📂 features:
|  |  |  |  |  |- 📂 Image:
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 imageTypes.ts
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 imageActions.ts
|  |  |  |  |  |  |  |- 📜 imageApi.ts
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 imageConstants.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 ImageReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageFlow.tsx
|  |  |  |  |  |  |  |- 📜 ImageTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 ImageParametersStep.tsx
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |  |- 📂 Tabular:
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 tabularTypes.ts
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 tabularActions.ts
|  |  |  |  |  |  |  |- 📜 tabularApi.ts
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 tabularConstants.ts
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 TabularDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularFlow.tsx
|  |  |  |  |  |  |  |- 📜 TabularReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularTrainspace.tsx
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |- 📂 types:
|  |  |  |  |  |- 📜 trainTypes.ts
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 trainspaceApi.ts
|  |  |  |  |  |- 📜 trainspaceSlice.ts
|  |  |  |  |- 📂 constants:
|  |  |  |  |  |- 📜 trainConstants.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 CreateTrainspace.tsx
|  |  |  |  |  |- 📜 DatasetStepLayout.tsx
|  |  |  |  |  |- 📜 TrainspaceLayout.tsx
|  |  |  |- 📂 LearnMod:
|  |  |  |  |- 📜 ClassCard.tsx
|  |  |  |  |- 📜 LearningModulesContent.tsx
|  |  |  |  |- 📜 ImageComponent.tsx
|  |  |  |  |- 📜 MCQuestion.tsx
|  |  |  |  |- 📜 FRQuestion.tsx
|  |  |  |  |- 📜 ModulesSideBar.tsx
|  |  |  |  |- 📜 Exercise.tsx
|  |  |  |- 📂 Feedback:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 feedbackApi.ts
|  |  |  |- 📂 OpenAi:
|  |  |  |  |- 📜 openAiUtils.ts
|  |  |  |- 📂 Dashboard:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 dashboardApi.ts
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TrainDoughnutChart.tsx
|  |  |  |  |  |- 📜 TrainBarChart.tsx
|  |  |  |  |  |- 📜 TrainDataGrid.tsx
|  |  |- 📂 common:
|  |  |  |- 📂 redux:
|  |  |  |  |- 📜 train.ts
|  |  |  |  |- 📜 userLogin.ts
|  |  |  |  |- 📜 hooks.ts
|  |  |  |  |- 📜 backendApi.ts
|  |  |  |  |- 📜 store.ts
|  |  |  |- 📂 utils:
|  |  |  |  |- 📜 firebase.ts
|  |  |  |  |- 📜 dateFormat.ts
|  |  |  |  |- 📜 dndHelpers.ts
|  |  |  |- 📂 styles:
|  |  |  |  |- 📜 Home.module.css
|  |  |  |  |- 📜 globals.css
|  |  |  |- 📂 components:
|  |  |  |  |- 📜 HtmlTooltip.tsx
|  |  |  |  |- 📜 NavBarMain.tsx
|  |  |  |  |- 📜 DlpTooltip.tsx
|  |  |  |  |- 📜 Footer.tsx
|  |  |  |  |- 📜 TitleText.tsx
|  |  |  |  |- 📜 EmailInput.tsx
|  |  |  |  |- 📜 ClientOnlyPortal.tsx
|  |  |  |  |- 📜 Spacer.tsx
|  |  |- 📂 pages:
|  |  |  |- 📂 train:
|  |  |  |  |- 📜 index.tsx
|  |  |  |  |- 📜 [train_space_id].tsx
|  |  |  |- 📜 LearnContent.tsx
|  |  |  |- 📜 login.tsx
|  |  |  |- 📜 settings.tsx
|  |  |  |- 📜 dashboard.tsx
|  |  |  |- 📜 forgot.tsx
|  |  |  |- 📜 feedback.tsx
|  |  |  |- 📜 learn.tsx
|  |  |  |- 📜 _document.tsx
|  |  |  |- 📜 wiki.tsx
|  |  |  |- 📜 about.tsx
|  |  |  |- 📜 _app.tsx
|  |  |- 📂 backend_outputs:
|  |  |  |- 📜 my_deep_learning_model.onnx : Last ONNX file output
|  |  |  |- 📜 model.pt : Last model.pt output
|  |  |  |- 📜 model.pkl
|  |  |- 📜 next-env.d.ts
|  |  |- 📜 iris.csv : Sample CSV data
|  |  |- 📜 GlobalStyle.ts
|  |  |- 📜 constants.ts
|  |- 📂 public:
|  |  |- 📂 images:
|  |  |  |- 📂 logos:
|  |  |  |  |- 📂 dlp_branding:
|  |  |  |  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |- 📜 dsgt-logo-dark.png
|  |  |  |  |- 📜 flask-logo.png
|  |  |  |  |- 📜 google.png
|  |  |  |  |- 📜 dsgt-logo-white-back.png
|  |  |  |  |- 📜 github.png
|  |  |  |  |- 📜 react-logo.png
|  |  |  |  |- 📜 pytorch-logo.png
|  |  |  |  |- 📜 aws-logo.png
|  |  |  |  |- 📜 dsgt-logo-light.png
|  |  |  |  |- 📜 pandas-logo.png
|  |  |  |  |- 📜 python-logo.png
|  |  |  |- 📂 wiki_images:
|  |  |  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |  |  |- 📜 batchnorm_diagram.png
|  |  |  |  |- 📜 dropout_diagram.png
|  |  |  |  |- 📜 conv2d.gif
|  |  |  |  |- 📜 conv2d2.gif
|  |  |  |  |- 📜 tanh_plot.png
|  |  |  |  |- 📜 maxpool2d.gif
|  |  |  |  |- 📜 tanh_equation.png
|  |  |  |  |- 📜 avgpool_maxpool.gif
|  |  |  |  |- 📜 sigmoid_equation.png
|  |  |  |- 📂 learn_mod_images:
|  |  |  |  |- 📜 sigmoidfunction.png
|  |  |  |  |- 📜 lossExampleEquation.png
|  |  |  |  |- 📜 ReLUactivation.png
|  |  |  |  |- 📜 robotImage.jpg
|  |  |  |  |- 📜 binarystepactivation.png
|  |  |  |  |- 📜 sigmoidactivation.png
|  |  |  |  |- 📜 lossExample.png
|  |  |  |  |- 📜 lossExampleTable.png
|  |  |  |  |- 📜 neuronWithEquation.png
|  |  |  |  |- 📜 neuralnet.png
|  |  |  |  |- 📜 neuron.png
|  |  |  |  |- 📜 LeakyReLUactivation.png
|  |  |  |  |- 📜 tanhactivation.png
|  |  |  |- 📜 demo_video.gif : GIF tutorial of a simple classification training session
|  |  |- 📜 dlp-logo.ico : DLP Logo
|  |  |- 📜 manifest.json : Default React file for choosing icon based on
|  |  |- 📜 robots.txt
|  |  |- 📜 index.html : Base HTML file that will be initially rendered
|  |- 📜 tsconfig.json
|  |- 📜 .eslintignore
|  |- 📜 next-env.d.ts
|  |- 📜 next.config.js
|  |- 📜 pnpm-lock.yaml
|  |- 📜 package.json
|  |- 📜 .eslintrc.json
|  |- 📜 jest.config.js
```

