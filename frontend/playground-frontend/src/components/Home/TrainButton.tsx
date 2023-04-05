import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { COLORS, ROUTE_DICT } from "../../constants";
import {
  validateParameter,
  validateTabularInputs,
  sendTabularJSON,
  validateImageInputs,
  sendPretrainedJSON,
  validatePretrainedInput,
  sendImageJSON,
  validateClassicalMLInput,
  sendClassicalMLJSON,
  validateObjectDetectionInput,
  sendObjectDetectionJSON,
  TrainParamsType,
} from "../helper_functions/TrainButtonFunctions";
import {
  sendEmail,
  uploadToBackend,
  train_and_output,
  JSONResponseType,
} from "../helper_functions/TalkWithBackend";
import { toast } from "react-toastify";
import { ModelLayer, ProblemType } from "../../settings";

interface TrainButtonPropTypes {
  addedLayers?: ModelLayer[];
  notification?: {
    email?: string;
    number?: string;
  };
  trainTransforms?: ModelLayer[];
  testTransforms?: ModelLayer[];
  transforms?: ModelLayer[];
  setDLPBackendResponse: React.Dispatch<
    React.SetStateAction<TrainResultsJSONResponseType | null>
  >;
  choice?: keyof typeof ROUTE_DICT;
  style?: object;
  problemType: ProblemType;
  usingDefaultDataset?: string;
  uploadFile?: File;
  customModelName?: string;
}
type TrainParamsWithPropsType = Omit<
  TrainButtonPropTypes,
  "trainTransforms" | "testTransforms" | "transforms" | "user_arch"
> &
  TrainParamsType;

interface DLResultsType {
  epoch: string;
  train_time: string;
  train_loss: string;
  test_loss: string;
  train_acc: string;
  "val/test acc": string;
}

interface TrainResultsJSONResponseType extends JSONResponseType {
  //TODO: make different types based on tabular, image, or classical
  auxiliary_outputs: {
    AUC_ROC_curve_data?: number[][][];
    category_list?: string[];
    confusion_matrix: number[][];
    numerical_category_list: number[];
    numerical_category_list_AUC?: number[];
    user_arch: string[];
  };
  dl_results: DLResultsType[];
}
const TrainButton = (props: TrainButtonPropTypes) => {
  const { uploadFile, setDLPBackendResponse, choice = "tabular" } = props;

  const [pendingResponse, setPendingResponse] = useState(false);
  const [result, setResult] = useState<TrainResultsJSONResponseType | null>(
    null
  );
  const [uploaded, setUploaded] = useState(false);
  const [trainParams, setTrainParams] = useState<{
    choice: keyof typeof ROUTE_DICT;
    paramList: TrainParamsWithPropsType;
  } | null>(null);
  const navigate = useNavigate();

  const reset = () => {
    setPendingResponse(false);
    setResult(null);
  };

  const make_obj_param_list = (
    obj_list: ModelLayer[] | undefined,
    source: "Model" | "Transforms" | "Train Transform" | "Test Transform"
  ) => {
    if (!obj_list) return; // ValidateInputs throw error in case of empty things. This is to prevent an unnecessary errors in case of creating a layer

    // making a array of relating methods (like "nn.Linear") with their parameters (in_feature, out_feature) by including all methods and their parameters to make something like:
    // ["nn.Linear(4, 10)", "nn.ReLU()", "nn.Linear(10, 3)", "nn.Softmax()"] OR
    // ["transforms.ToTensor()", "transforms.RandomHorizontalFlip(0.8)"]

    const user_arch = [];
    for (let i = 0; i < obj_list.length; i++) {
      const obj_list_item = obj_list[i];
      const parameters = obj_list_item.parameters;
      let parameter_call_input = "";
      const is_transform_type =
        obj_list_item.transform_type &&
        obj_list_item.transform_type === "functional";
      const parameters_to_be_added = Array(Object.keys(parameters).length);
      for (const v of Object.values(parameters)) {
        if (!validateParameter(source, i, v)) {
          reset();
          return false;
        }
        const parameter_value =
          v.parameter_type === "number" || v.parameter_type === "tuple"
            ? v.value
            : `'${v.value}'`;
        parameters_to_be_added[v.index] = `${v.kwarg ?? ""}${parameter_value}`;
      }
      if (is_transform_type) {
        parameter_call_input += "img, ";
      }
      parameters_to_be_added.forEach((e) => {
        parameter_call_input += e + ",";
      });
      // removing the last ','
      parameter_call_input = parameter_call_input.slice(0, -1);

      let callback = `${obj_list_item.object_name}(${parameter_call_input})`;
      if (is_transform_type) {
        callback = "transforms.Lambda(lambda img: " + callback + ")";
      }
      user_arch.push(callback);
    }
    return user_arch;
  };

  const functionMap: { [trainType: string]: unknown[] } = {
    tabular: [validateTabularInputs, sendTabularJSON],
    image: [validateImageInputs, sendImageJSON],
    pretrained: [validatePretrainedInput, sendPretrainedJSON],
    classicalml: [validateClassicalMLInput, sendClassicalMLJSON],
    objectdetection: [validateObjectDetectionInput, sendObjectDetectionJSON],
  };

  const validateInputs = (user_arch: string[]) => {
    let alertMessage = "";
    alertMessage = (
      functionMap[choice][0] as (
        user_arch: string[],
        props: TrainButtonPropTypes
      ) => string
    )(user_arch, props);

    if (alertMessage.length === 0) return true;
    toast.error(alertMessage);
    return false;
  };

  const onClick = async () => {
    setPendingResponse(true);
    setDLPBackendResponse(null);
    const user_arch = make_obj_param_list(props.addedLayers, "Model");
    if (user_arch === false) return;

    let trainTransforms: boolean | string[] | undefined;
    let testTransforms: boolean | string[] | undefined;
    let transforms: boolean | string[] | undefined;
    if (props.trainTransforms) {
      trainTransforms = make_obj_param_list(
        props.trainTransforms,
        "Train Transform"
      );
      if (trainTransforms === false) return;
    }
    if (props.testTransforms) {
      testTransforms = make_obj_param_list(
        props.testTransforms,
        "Test Transform"
      );
      if (testTransforms === false) return;
    }
    if (props.transforms) {
      transforms = make_obj_param_list(props.transforms, "Transforms");
      if (transforms === false) return;
    }

    if (user_arch && !validateInputs(user_arch)) {
      setPendingResponse(false);
      return;
    }

    const paramList: TrainParamsWithPropsType = {
      ...props,
      trainTransforms: trainTransforms as string[],
      testTransforms: testTransforms as string[],
      transforms: transforms as string[],
      user_arch: user_arch as string[],
    };

    if (
      (choice === "image" && !props.usingDefaultDataset) ||
      choice === "objectdetection"
    ) {
      const formData = new FormData();
      if (uploadFile) {
        formData.append("file", uploadFile);
        await uploadToBackend(formData);
      }
    }
    const trainState = await train_and_output(
      choice,
      (
        functionMap[choice][1] as (paramList: TrainParamsWithPropsType) => {
          [key: string]: unknown;
        }
      )(paramList)
    );
    if (process.env.REACT_APP_MODE === "prod") {
      if (trainState.success) toast.success(trainState.message);
      else toast.error(trainState.message);

      navigate("/dashboard");
    } else {
      setResult(trainState as TrainResultsJSONResponseType);
    }
  };

  useEffect(() => {
    if (uploaded && trainParams) {
      train_and_output(
        trainParams.choice,
        (
          functionMap[trainParams.choice][1] as (
            paramList: TrainParamsWithPropsType
          ) => { [key: string]: unknown }
        )(trainParams.paramList)
      );
      setUploaded(false);
      setTrainParams(null);
    }
  }, [uploaded, trainParams]);

  useEffect(() => {
    if (result) {
      if (result.success) {
        if (props.notification?.email) {
          sendEmail(props.notification.email, props.problemType);
        }
        toast.success(
          choice === "objectdetection"
            ? "Detection successful! Scroll to see results!"
            : "Training successful! Scroll to see results!"
        );
      } else if (result.message) {
        toast.error("Training failed. Check output traceback message");
      } else {
        toast.error("Training failed. Check your inputs");
      }
      setDLPBackendResponse(result);
      reset();
    }
  }, [result]);

  return (
    <>
      <button
        id="train-button"
        className="btn btn-primary"
        style={{
          backgroundColor: pendingResponse ? COLORS.disabled : undefined,
          cursor: pendingResponse ? "wait" : "pointer",
        }}
        onClick={onClick}
        disabled={pendingResponse}
      >
        {choice === "objectdetection" ? "Run!" : "Train!"}
      </button>
      {pendingResponse ? <div className="loader" /> : null}
    </>
  );
};

export default TrainButton;
