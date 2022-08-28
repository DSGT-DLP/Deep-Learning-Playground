/**
 * This file's puropose is to generalise the methods of TrainButton (focusing on Tabular, Image, and Pretrained models)
 *
 */

// TABULAR
export const validateTabularInputs = (user_arch, ...args) => {
  args = args[0];
  let alertMessage = "";
  if (!user_arch?.length) alertMessage += "At least one layer must be added. ";
  if (!args.criterion) alertMessage += "A criterion must be specified. ";
  if (!args.optimizerName)
    alertMessage += "An optimizer name must be specified. ";
  if (!args.problemType) alertMessage += "A problem type must be specified. ";
  if (!args.usingDefaultDataset) {
    if (!args.targetCol || !args.features?.length) {
      alertMessage +=
        "Must specify an input file, target, and features if not selecting default dataset. ";
    }
    for (let i = 0; i < args.features.length; i++) {
      if (args.targetCol === args.features[i]) {
        alertMessage +=
          "A column that is selected as the target column cannot also be a feature column. ";
        break;
      }
    }
    if (!args.csvDataInput && !args.fileURL) {
      alertMessage +=
        "Must specify an input file either from local storage or from an internet URL. ";
    }
  }
  if (args.batchSize < 2) alertMessage += "Batch size cannot be less than 2";
  return alertMessage;
};

export const sendTabularJSON = (...args) => {
  args = args[0];

  const csvDataStr = JSON.stringify(args.csvDataInput);

  return {
    user_arch: args.user_arch,
    criterion: args.criterion,
    optimizer_name: args.optimizerName,
    problem_type: args.problemType,
    target: args.targetCol != null ? args.targetCol : null,
    features: args.features ? args.features : null,
    using_default_dataset: args.usingDefaultDataset
      ? args.usingDefaultDataset
      : null,
    test_size: args.testSize,
    epochs: args.epochs,
    batch_size: args.batchSize,
    shuffle: args.shuffle,
    csv_data: csvDataStr,
    file_URL: args.fileURL,
    email: args.email,
  };
};

// IMAGE
export const validateImageInputs = (user_arch, ...args) => {
  args = args[0];
  let alertMessage = "";
  if (!user_arch?.length) alertMessage += "At least one layer must be added. ";
  if (!args.criterion) alertMessage += "A criterion must be specified. ";
  if (!args.optimizerName)
    alertMessage += "An optimizer name must be specified. ";
  if (args.batchSize < 2) alertMessage += "Batch size cannot be less than 2";
  if (!args.dataUploaded && !args.usingDefaultDataset)
    alertMessage += "Please specify a valid data from default or upload";
  // can easily add a epoch limit

  return alertMessage;
};

export const sendImageJSON = (...args) => {
  args = args[0];

  return {
    user_arch: args.user_arch,
    criterion: args.criterion,
    optimizer_name: args.optimizerName,
    using_default_dataset: args.usingDefaultDataset
      ? args.usingDefaultDataset
      : null,
    epochs: args.epochs,
    batch_size: args.batchSize,
    shuffle: args.shuffle,
    file_URL: args.fileURL,
    train_transform: args.trainTransforms,
    test_transform: args.testTransforms,
    email: args.email ? args.email : null,
  };
};

// PRETRAINED

export const validatePretrainedInput = (user_arch, ...args) => {
  args = args[0];
  let alertMessage = "";
  if (!args.modelName) alertMessage += "A model name must be specified.";
  if (!args.criterion) alertMessage += "A criterion must be specified. ";
  if (!args.optimizerName)
    alertMessage += "An optimizer name must be specified. ";
  if (!args.usingDefaultDataset) {
    if (!args.dataInput && !args.fileURL) {
      alertMessage +=
        "Must specify an input file either from local storage or from an internet URL. ";
    }
  }

  return alertMessage;
};

export const sendPretrainedJSON = (...args) => {
  args = args[0];

  return {
    model_name: args.modelName,
    criterion: args.criterion,
    optimizer_name: args.optimizerName,
    using_default_dataset: args.usingDefaultDataset
      ? args.usingDefaultDataset
      : null,
    epochs: args.epochs,
    batch_size: args.batchSize,
    shuffle: args.shuffle,
    file_URL: args.fileURL,
    train_transform: args.trainTransforms,
    test_transform: args.testTransforms,
    email: args.email,
  };
};
