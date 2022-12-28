import { toast } from "react-toastify";

/**
 * This file's puropose is to generalise the methods of TrainButton (focusing on Tabular, Image, and Pretrained models)
 *
 */
const tupleRegex = /^\(([1-9]{1}[0-9]*), ?([1-9]{1}[0-9]*)\)$/;

export const validateParameter = (source, index, parameter) => {
  const { parameter_name, min, max, parameter_type } = parameter;
  let { value } = parameter;
  if (parameter_name === "(H, W)") {
    if (tupleRegex.test(value)) {
      const result = value.match(tupleRegex);
      const H = result[1].valueOf();
      const W = result[2].valueOf();

      if (H < min || H > max) {
        toast.error(
          `${source} Layer ${
            index + 1
          }: H not an integer in range [${min}, ${max}]`
        );
        return false;
      } else if (W < min || W > max) {
        toast.error(
          `${source} Layer ${
            index + 1
          }: W not an integer in range [${min}, ${max}]`
        );
        return false;
      }
      return true;
    }
    toast.error(
      `${source} Layer ${
        index + 1
      }: ${parameter_name} not of appropriate format: (H, W)`
    );
  } else {
    if (parameter_type !== "number") return true;

    if (min == null && max == null) return true;

    if (min == null && value <= max) return true;

    if (value >= min && max == null) return true;

    if (value >= min && value <= max) return true;

    toast.error(
      `${source} Layer ${
        index + 1
      }: ${parameter_name} not an integer in range [${min}, ${max}]`
    );
  }
  return false;
};

// TABULAR
export const validateTabularInputs = (user_arch, ...args) => {
  args = args[0];
  let alertMessage = "";
  if (!user_arch?.length) alertMessage += "At least one layer must be added. ";
  if (!args.customModelName)
    alertMessage += "Custom model name must be specified. ";
  if (!args.criterion) alertMessage += "A criterion must be specified. ";
  if (!args.optimizerName)
    alertMessage += "An optimizer name must be specified. ";
  if (!args.problemType) alertMessage += "A problem type must be specified. ";
  if (!args.usingDefaultDataset) {
    if (!args.targetCol || !args.features?.length) {
      alertMessage +=
        "Must specify an input file, target, and features if not selecting default dataset. ";
    }
    for (let i = 0; i < args.features?.length; i++) {
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
    custom_model_name: args.customModelName,
  };
};

// IMAGE
export const validateImageInputs = (user_arch, ...args) => {
  args = args[0];
  let alertMessage = "";
  if (!user_arch?.length) alertMessage += "At least one layer must be added. ";
  if (!args.customModelName)
    alertMessage += "Custom model name must be specified. ";
  if (!args.criterion) alertMessage += "A criterion must be specified. ";
  if (!args.optimizerName)
    alertMessage += "An optimizer name must be specified. ";
  if (args.batchSize < 2) alertMessage += "Batch size cannot be less than 2";
  if (!args.uploadFile && !args.usingDefaultDataset)
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
    custom_model_name: args.customModelName,
  };
};

// PRETRAINED

export const validatePretrainedInput = (user_arch, ...args) => {
  args = args[0];

  let alertMessage = "";
  console.log(args.beginnerMode);
  if (!args.customModelName)
    alertMessage += "Custom model name must be specified. ";
  if (!args.modelName && !args.beginnerMode)
    alertMessage += "A model name must be specified.";
  if (!args.criterion) alertMessage += "A criterion must be specified. ";
  if (!args.optimizerName)
    alertMessage += "An optimizer name must be specified. ";
  if (!args.uploadFile && !args.usingDefaultDataset) {
    alertMessage +=
      "Must specify an input file either from local storage or from an internet URL. ";
  }

  return alertMessage;
};

export const sendPretrainedJSON = (...args) => {
  args = args[0];

  return {
    model_name: args.modelName ? args.modelName : "resnet18",
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
    custom_model_name: args.customModelName,
  };
};

//Classical ML
export const validateClassicalMLInput = (user_arch, ...args) => {

  args = args[0];
  let alertMessage = "";
  if (!args.problemType) alertMessage += "A problem type must be specified. ";
  if (!args.usingDefaultDataset) {
    if (!args.targetCol || !args.features?.length) {
      alertMessage +=
        "Must specify an input file, target, and features if not selecting default dataset. ";
    }
    for (let i = 0; i < args.features?.length; i++) {
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
  return alertMessage;
};

export const sendClassicalMLJSON = (...args) => {
  args = args[0];

  const csvDataStr = JSON.stringify(args.csvDataInput);

  return {
    user_arch: args.user_arch,
    problem_type: args.problemType,
    target: args.targetCol != null ? args.targetCol : null,
    features: args.features ? args.features : null,
    using_default_dataset: args.usingDefaultDataset
      ? args.usingDefaultDataset
      : null,
    test_size: args.testSize,
    shuffle: args.shuffle,
    csv_data: csvDataStr,
    file_URL: args.fileURL,
    email: args.email,
  };
};

export const validateObjectDetectionInput = (user_arch, ...args) => {
  args = args[0];
  let alertMessage = "";
  if (!args.uploadFile)
    alertMessage += "Must specify an input file from local storage. ";
  if (!args.problemType) alertMessage += "A problem type must be specified. ";
  return alertMessage;
};

export const sendObjectDetectionJSON = (...args) => {
  args = args[0];
  return {
    problem_type: args.problemType,
  };
};
