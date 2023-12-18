# Architecture

## Backend Architecture

```
📦 backend
|  |- 📂 ml:
|  |  |- 📜 __init__.py
|  |  |- 📜 ml_model_parser.py
|  |  |- 📜 ml_trainer.py : train a classical machine learning learning model on the dataset
|  |- 📂 common:
|  |  |- 📜 ai_drive.py
|  |  |- 📜 preprocessing.py
|  |  |- 📜 email_notifier.py : Endpoint to send email notification of training results via API Gateway + AWS SES
|  |  |- 📜 default_datasets.py : store logic to load in default datasets from scikit-learn
|  |  |- 📜 dataset.py : read in the dataset through URL or file upload
|  |  |- 📜 constants.py : list of helpful constants
|  |  |- 📜 utils.py : utility functions that could be helpful
|  |  |- 📜 __init__.py
|  |  |- 📜 loss_functions.py : loss function enum
|  |  |- 📜 kernel.py
|  |  |- 📜 optimizer.py : what optimizer to use (ie: SGD or Adam for now)
|  |- 📂 dl:
|  |  |- 📜 detection.py
|  |  |- 📜 dl_model_parser.py : parse the user specified pytorch model
|  |  |- 📜 dl_eval.py : Evaluation functions for deep learning models in Pytorch (eg: accuracy, loss, etc)
|  |  |- 📜 __init__.py
|  |  |- 📜 dl_model.py : torch model based on user specifications from drag and drop
|  |  |- 📜 dl_trainer.py : train a deep learning model on the dataset
|  |- 📂 aws_helpers:
|  |  |- 📂 dynamo_db_utils:
|  |  |  |- 📜 trainspace_db.py
|  |  |  |- 📜 userprogress_db.py
|  |  |  |- 📜 constants.py : list of helpful constants
|  |  |  |- 📜 DynamoUnitTests.md
|  |  |  |- 📜 dynamo_db_utils.py
|  |  |- 📜 __init__.py
|  |- 📜 app.py : run the backend (entrypoint script)
|  |- 📜 poetry.lock
|  |- 📜 middleware.py
|  |- 📜 __init__.py
|  |- 📜 data.csv : data csv file for use in the playground
|  |- 📜 epoch_times.csv
|  |- 📜 pyproject.toml
```

## Frontend Architecture

```
📦 frontend
|  |- 📂 layer_docs:
|  |  |- 📜 ReLU.md : Doc for ReLU later
|  |  |- 📜 Linear.md : Doc for Linear layer
|  |  |- 📜 Softmax.md : Doc for Softmax layer
|  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |- 📂 public:
|  |  |- 📂 images:
|  |  |  |- 📂 logos:
|  |  |  |  |- 📂 dlp_branding:
|  |  |  |  |  |- 📜 dlp-logo.png : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |  |- 📜 dlp-logo.svg : DLP Logo, duplicate of files in public, but essential as the frontend can't read public
|  |  |  |  |- 📜 flask-logo.png
|  |  |  |  |- 📜 dsgt-logo-dark.png
|  |  |  |  |- 📜 dsgt-logo-light.png
|  |  |  |  |- 📜 python-logo.png
|  |  |  |  |- 📜 github.png
|  |  |  |  |- 📜 google.png
|  |  |  |  |- 📜 aws-logo.png
|  |  |  |  |- 📜 react-logo.png
|  |  |  |  |- 📜 pandas-logo.png
|  |  |  |  |- 📜 pytorch-logo.png
|  |  |  |  |- 📜 dsgt-logo-white-back.png
|  |  |  |- 📂 wiki_images:
|  |  |  |  |- 📜 avgpool_maxpool.gif
|  |  |  |  |- 📜 dropout_diagram.png
|  |  |  |  |- 📜 tanh_equation.png
|  |  |  |  |- 📜 maxpool2d.gif
|  |  |  |  |- 📜 conv2d2.gif
|  |  |  |  |- 📜 sigmoid_equation.png
|  |  |  |  |- 📜 softmax_equation.png : PNG file of Softmax equation
|  |  |  |  |- 📜 conv2d.gif
|  |  |  |  |- 📜 batchnorm_diagram.png
|  |  |  |  |- 📜 tanh_plot.png
|  |  |  |- 📂 learn_mod_images:
|  |  |  |  |- 📜 sigmoidactivation.png
|  |  |  |  |- 📜 sigmoidfunction.png
|  |  |  |  |- 📜 lossExampleTable.png
|  |  |  |  |- 📜 lossExample.png
|  |  |  |  |- 📜 LeakyReLUactivation.png
|  |  |  |  |- 📜 robotImage.jpg
|  |  |  |  |- 📜 tanhactivation.png
|  |  |  |  |- 📜 ReLUactivation.png
|  |  |  |  |- 📜 neuron.png
|  |  |  |  |- 📜 binarystepactivation.png
|  |  |  |  |- 📜 lossExampleEquation.png
|  |  |  |  |- 📜 neuronWithEquation.png
|  |  |  |  |- 📜 neuralnet.png
|  |  |  |- 📜 demo_video.gif : GIF tutorial of a simple classification training session
|  |  |- 📜 manifest.json : Default React file for choosing icon based on
|  |  |- 📜 robots.txt
|  |  |- 📜 index.html : Base HTML file that will be initially rendered
|  |  |- 📜 dlp-logo.ico : DLP Logo
|  |- 📂 src:
|  |  |- 📂 features:
|  |  |  |- 📂 LearnMod:
|  |  |  |  |- 📜 LearningModulesContent.tsx
|  |  |  |  |- 📜 FRQuestion.tsx
|  |  |  |  |- 📜 MCQuestion.tsx
|  |  |  |  |- 📜 Exercise.tsx
|  |  |  |  |- 📜 ModulesSideBar.tsx
|  |  |  |  |- 📜 ImageComponent.tsx
|  |  |  |  |- 📜 ClassCard.tsx
|  |  |  |- 📂 Dashboard:
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 TrainBarChart.tsx
|  |  |  |  |  |- 📜 TrainDoughnutChart.tsx
|  |  |  |  |  |- 📜 TrainDataGrid.tsx
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 dashboardApi.ts
|  |  |  |- 📂 OpenAi:
|  |  |  |  |- 📜 openAiUtils.ts
|  |  |  |- 📂 Feedback:
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 feedbackApi.ts
|  |  |  |- 📂 Train:
|  |  |  |  |- 📂 components:
|  |  |  |  |  |- 📜 CreateTrainspace.tsx
|  |  |  |  |  |- 📜 TrainspaceLayout.tsx
|  |  |  |  |  |- 📜 DatasetStepLayout.tsx
|  |  |  |  |- 📂 features:
|  |  |  |  |  |- 📂 Image:
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 ImageFlow.tsx
|  |  |  |  |  |  |  |- 📜 ImageDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 ImageTrainspace.tsx
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 imageTypes.ts
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 imageConstants.ts
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 imageApi.ts
|  |  |  |  |  |  |  |- 📜 imageActions.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |  |- 📂 Tabular:
|  |  |  |  |  |  |- 📂 components:
|  |  |  |  |  |  |  |- 📜 TabularTrainspace.tsx
|  |  |  |  |  |  |  |- 📜 TabularDatasetStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularParametersStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularReviewStep.tsx
|  |  |  |  |  |  |  |- 📜 TabularFlow.tsx
|  |  |  |  |  |  |- 📂 types:
|  |  |  |  |  |  |  |- 📜 tabularTypes.ts
|  |  |  |  |  |  |- 📂 constants:
|  |  |  |  |  |  |  |- 📜 tabularConstants.ts
|  |  |  |  |  |  |- 📂 redux:
|  |  |  |  |  |  |  |- 📜 tabularApi.ts
|  |  |  |  |  |  |  |- 📜 tabularActions.ts
|  |  |  |  |  |  |- 📜 index.ts
|  |  |  |  |- 📂 types:
|  |  |  |  |  |- 📜 trainTypes.ts
|  |  |  |  |- 📂 constants:
|  |  |  |  |  |- 📜 trainConstants.ts
|  |  |  |  |- 📂 redux:
|  |  |  |  |  |- 📜 trainspaceApi.ts
|  |  |  |  |  |- 📜 trainspaceSlice.ts
|  |  |- 📂 pages:
|  |  |  |- 📂 train:
|  |  |  |  |- 📜 [train_space_id].tsx
|  |  |  |  |- 📜 index.tsx
|  |  |  |- 📜 forgot.tsx
|  |  |  |- 📜 LearnContent.tsx
|  |  |  |- 📜 _document.tsx
|  |  |  |- 📜 settings.tsx
|  |  |  |- 📜 feedback.tsx
|  |  |  |- 📜 wiki.tsx
|  |  |  |- 📜 _app.tsx
|  |  |  |- 📜 about.tsx
|  |  |  |- 📜 login.tsx
|  |  |  |- 📜 dashboard.tsx
|  |  |  |- 📜 learn.tsx
|  |  |- 📂 backend_outputs:
|  |  |  |- 📜 model.pt : Last model.pt output
|  |  |  |- 📜 model.pkl
|  |  |  |- 📜 my_deep_learning_model.onnx : Last ONNX file output
|  |  |- 📂 common:
|  |  |  |- 📂 components:
|  |  |  |  |- 📜 NavBarMain.tsx
|  |  |  |  |- 📜 ClientOnlyPortal.tsx
|  |  |  |  |- 📜 HtmlTooltip.tsx
|  |  |  |  |- 📜 Spacer.tsx
|  |  |  |  |- 📜 EmailInput.tsx
|  |  |  |  |- 📜 Footer.tsx
|  |  |  |  |- 📜 DlpTooltip.tsx
|  |  |  |  |- 📜 TitleText.tsx
|  |  |  |- 📂 styles:
|  |  |  |  |- 📜 globals.css
|  |  |  |  |- 📜 Home.module.css
|  |  |  |- 📂 utils:
|  |  |  |  |- 📜 dndHelpers.ts
|  |  |  |  |- 📜 dateFormat.ts
|  |  |  |  |- 📜 firebase.ts
|  |  |  |- 📂 redux:
|  |  |  |  |- 📜 train.ts
|  |  |  |  |- 📜 userLogin.ts
|  |  |  |  |- 📜 store.ts
|  |  |  |  |- 📜 hooks.ts
|  |  |  |  |- 📜 backendApi.ts
|  |  |- 📜 next-env.d.ts
|  |  |- 📜 iris.csv : Sample CSV data
|  |  |- 📜 constants.ts
|  |  |- 📜 GlobalStyle.ts
|  |- 📜 next-env.d.ts
|  |- 📜 package.json
|  |- 📜 next.config.js
|  |- 📜 .eslintignore
|  |- 📜 .eslintrc.json
|  |- 📜 tsconfig.json
|  |- 📜 pnpm-lock.yaml
|  |- 📜 jest.config.js
```

