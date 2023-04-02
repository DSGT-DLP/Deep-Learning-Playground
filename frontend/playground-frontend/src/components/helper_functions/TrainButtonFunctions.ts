import { toast } from "react-toastify";
import {
  Criterion,
  DefaultDatasetType,
  LayerParameter,
  ModelLayer,
  ProblemType,
} from "../../settings";
import { CSVInputDataRowType } from "../Home/CSVInputFile";

/**
 * This file's puropose is to generalise the methods of TrainButton (focusing on Tabular, Image, and Pretrained models)
 *
 */
const tupleRegex = /^\(([0-9]{1}[0-9]*), ?([0-9]{1}[0-9]*)\)$/;

export const validateParameter = (
  source: "Model" | "Transforms" | "Train Transform" | "Test Transform",
  index: number,
  parameter: LayerParameter
) => {
  if (parameter.value) {
    if (parameter.parameter_type === "tuple") {
      const { parameter_name, min, max, value } =
        parameter as LayerParameter<"tuple">;
      if (value) {
        if (tupleRegex.test(value)) {
          const result = value.match(tupleRegex);
          if (result) {
            const H = parseInt(result[1].valueOf());
            const W = parseInt(result[2].valueOf());

            if (H < min || H > max) {
              toast.error(
                `${source} Layer ${
                  index + 1
                }: X not an integer in range [${min}, ${max}]`
              );
              return false;
            } else if (W < min || W > max) {
              toast.error(
                `${source} Layer ${
                  index + 1
                }: Y not an integer in range [${min}, ${max}]`
              );
              return false;
            }
            return true;
          }
        }
      }
      toast.error(
        `${source} Layer ${
          index + 1
        }: ${parameter_name} not of appropriate format: (X, Y)`
      );
    } else {
      if (parameter.parameter_type !== "number") return true;
      const { parameter_name, min, max, value } =
        parameter as LayerParameter<"number">;
      if (value) {
        if (min == null && max == null) return true;

        if (min == null && value <= max) return true;

        if (value >= min && max == null) return true;

        if (value >= min && value <= max) return true;
      }

      toast.error(
        `${source} Layer ${
          index + 1
        }: ${parameter_name} not an integer in range [${min}, ${max}]`
      );
    }
  }
  toast.error(
    `${source} Layer ${index + 1}: ${
      parameter.parameter_name
    } value not specified`
  );
  return false;
};

interface FeatureType {
  label: string;
  value: number;
}

interface TrainParamsType {
  user_arch: ModelLayer[];
  trainTransforms: string[];
  testTransforms: string[];
  transforms: string[];
}
// TABULAR
interface TabularInputType {
  customModelName: string | null;
  criterion: Criterion | null;
  optimizerName: string | null;
  problemType: ProblemType | null;
  usingDefaultDataset: DefaultDatasetType;
  targetCol: FeatureType | null;
  features: FeatureType[] | null;
  csvDataInput: CSVInputDataRowType[] | null;
  fileURL: string | null;
  shuffle: boolean;
  epochs: number;
  testSize: number;
  notification: {
    email?: string;
    phoneNumber?: string;
  };
  batchSize: number;
}
export const validateTabularInputs = (
  user_arch: ModelLayer[],
  args: TabularInputType
) => {
  if (!user_arch?.length) return "At least one layer must be added. ";
  if (!args.customModelName) return "Custom model name must be specified. ";
  if (!args.criterion) return "A criterion must be specified. ";
  if (!args.optimizerName) return "An optimizer name must be specified. ";
  if (!args.problemType) return "A problem type must be specified. ";
  if (!args.usingDefaultDataset) {
    if (!args.targetCol || !args.features?.length) {
      return "Must specify an input file, target, and features if not selecting default dataset. ";
    }
    for (let i = 0; i < args.features?.length; i++) {
      if (args.targetCol === args.features[i]) {
        return "A column that is selected as the target column cannot also be a feature column. ";
      }
    }
    if (!args.csvDataInput && !args.fileURL) {
      return "Must specify an input file either from local storage or from an internet URL. ";
    }
  }
  if (args.batchSize < 2) return "Batch size cannot be less than 2";
  return "";
};

export const sendTabularJSON = (args: TabularInputType & TrainParamsType) => {
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
    notification: args.notification,
    custom_model_name: args.customModelName,
    data_source: "TABULAR",
  };
};

interface ImageInputType {
  batchSize: number;
  criterion: Criterion | null;
  shuffle: boolean;
  epochs: number;
  optimizerName: string | null;
  addedLayers: ModelLayer[];
  usingDefaultDataset: DefaultDatasetType;
  trainTransforms: string[];
  testTransforms: string[];
  uploadFile: File;
  customModelName: string | null;
}
// IMAGE
export const validateImageInputs = (
  user_arch: ModelLayer[],
  args: ImageInputType
) => {
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

export const sendImageJSON = (args: ImageInputType & TrainParamsType) => {
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
    //file_URL: args.fileURL,
    train_transform: args.trainTransforms,
    test_transform: args.testTransforms,
    //email: args.email ? args.email : null,
    custom_model_name: args.customModelName,
    data_source: "IMAGE",
  };
};

// PRETRAINED

interface PreTrainedInputType {
  customModelName: string | null;
  modelName: string | null;
  criterion: Criterion | null;
  optimizerName: string | null;
  problemType: ProblemType | null;
  usingDefaultDataset: DefaultDatasetType;
  targetCol: FeatureType | null;
  features: FeatureType[] | null;
  csvDataInput: CSVInputDataRowType[] | null;
  dataInput: CSVInputDataRowType[] | null;
  fileURL: string | null;
  shuffle: boolean;
  epochs: number;
  testSize: number;
  email?: string;
  batchSize: number;
}
export const validatePretrainedInput = (
  user_arch: ModelLayer[],
  args: PreTrainedInputType
) => {
  let alertMessage = "";
  if (!args.customModelName)
    alertMessage += "Custom model name must be specified. ";
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

export const sendPretrainedJSON = (
  args: PreTrainedInputType & TrainParamsType
) => {
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
    custom_model_name: args.customModelName,
    data_source: "PRETRAINED",
  };
};

//Classical ML
interface ClassicalMLModelInputType {
  shuffle: boolean;
  problemType: ProblemType | null;
  addedLayers: ModelLayer[];
  targetCol: FeatureType;
  features: FeatureType[];
  usingDefaultDataset: DefaultDatasetType;
  customModelName: string | null;
  csvDataInput: CSVInputDataRowType[] | null;
}
export const validateClassicalMLInput = (
  user_arch: ModelLayer[],
  args: ClassicalMLModelInputType
) => {
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
    if (!args.csvDataInput /* && !args.fileURL*/) {
      alertMessage +=
        "Must specify an input file either from local storage or from an internet URL. ";
    }
  }
  return alertMessage;
};

export const sendClassicalMLJSON = (
  args: ClassicalMLModelInputType & TrainParamsType
) => {
  const csvDataStr = JSON.stringify(args.csvDataInput);

  return {
    user_arch: args.user_arch,
    problem_type: args.problemType,
    target: args.targetCol != null ? args.targetCol : null,
    features: args.features ? args.features : null,
    using_default_dataset: args.usingDefaultDataset
      ? args.usingDefaultDataset
      : null,
    //test_size: args.testSize,
    shuffle: args.shuffle,
    csv_data: csvDataStr,
    //file_URL: args.fileURL,
    //email: args.email,
    data_source: "CLASSICAL_ML",
  };
};

interface ObjectDetectionInputType {
  uploadFile: File;
  problemType: string;
  detectionType: string;
}
export const validateObjectDetectionInput = (
  user_arch: ModelLayer[],
  args: ObjectDetectionInputType
) => {
  let alertMessage = "";
  if (!args.uploadFile)
    alertMessage += "Must specify an input file from local storage. ";
  if (args.detectionType === "rekognition" && !args.problemType)
    alertMessage += "A problem type must be specified. ";
  if (!args.detectionType)
    alertMessage += "A detection type must be specified. ";
  return alertMessage;
};

export const sendObjectDetectionJSON = (
  args: ObjectDetectionInputType & TrainParamsType
) => {
  return {
    problem_type: args.problemType != null ? args.problemType : null,
    detection_type: args.detectionType,
    transforms: args.transforms,
    data_source: "OBJECT_DETECTION",
  };
};
