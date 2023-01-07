import React, { useState, useEffect } from "react";
import PropTypes from "prop-types";
import { COLORS } from "../../constants";
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
} from "../helper_functions/TrainButtonFunctions";
import {
  sendEmail,
  uploadToBackend,
  train_and_output,
} from "../helper_functions/TalkWithBackend";
import { toast } from "react-toastify";

const TrainButton = (props) => {
  const { uploadFile, setDLPBackendResponse, choice = "tabular" } = props;

  const [pendingResponse, setPendingResponse] = useState(false);
  const [result, setResult] = useState(null);
  const [uploaded, setUploaded] = useState(false);
  const [trainParams, setTrainParams] = useState(null);

  const reset = () => {
    setPendingResponse(false);
    setResult(null);
  };

  const make_obj_param_list = (obj_list, source) => {
    if (!obj_list) return; // ValidateInputs throw error in case of empty things. This is to prevent an unnecessary errors in case of creating a layer

    // making a array of relating methods (like "nn.Linear") with their parameters (in_feature, out_feature) by including all methods and their parameters to make something like:
    // ["nn.Linear(4, 10)", "nn.ReLU()", "nn.Linear(10, 3)", "nn.Softmax()"] OR
    // ["transforms.ToTensor()", "transforms.RandomHorizontalFlip(0.8)"]

    const user_arch = [];
    for (let i = 0; i < obj_list.length; i++) {
      const obj_list_item = obj_list[i];
      const parameters = obj_list_item.parameters;
      let parameter_call_input = "";
      let transform_type = obj_list_item.transform_type && obj_list_item.transform_type === "functional";
      const parameters_to_be_added = Array(Object.keys(parameters).length);
      for (const v of Object.values(parameters)) {
        if (!validateParameter(source, i, v)) {
          reset();
          return false;
        }
        const parameter_value =
          v.parameter_type === "number" || v.parameter_type === "tuple" ? v.value : `'${v.value}'`;
        parameters_to_be_added[v.index] = `${v.kwarg ?? ""}${parameter_value}`;
      }
      if (transform_type) {
        parameter_call_input += "img, ";
      }
      parameters_to_be_added.forEach((e) => {
        parameter_call_input += e + ",";
      });
      // removing the last ','
      parameter_call_input = parameter_call_input.slice(0, -1);

      let callback = `${obj_list_item.object_name}(${parameter_call_input})`;
      if (transform_type) {
        callback = "transforms.Lambda(lambda img: " + callback + ")";
      }
      user_arch.push(callback);
    }
    return user_arch;
  };

  const functionMap = {
    tabular: [validateTabularInputs, sendTabularJSON],
    image: [validateImageInputs, sendImageJSON],
    pretrained: [validatePretrainedInput, sendPretrainedJSON],
    classicalml: [validateClassicalMLInput, sendClassicalMLJSON],
    objectdetection: [validateObjectDetectionInput, sendObjectDetectionJSON],
  };

  const validateInputs = (user_arch) => {
    let alertMessage = "";
    alertMessage = functionMap[choice][0](user_arch, props);
    if (alertMessage.length === 0) return true;
    toast.error(alertMessage);
    return false;
  };

  const onClick = async () => {
    setPendingResponse(true);
    setDLPBackendResponse(null);

    const user_arch = make_obj_param_list(props.addedLayers, "Model");
    if (user_arch === false) return;

    let trainTransforms = 0;
    let testTransforms = 0;
    let transforms = 0;
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
      transforms = make_obj_param_list(
        props.transforms,
        "Transforms"
      );
      if (transforms === false) return;
    }

    if (!validateInputs(user_arch)) {
      setPendingResponse(false);
      return;
    }

    const paramList = { ...props, trainTransforms, testTransforms, transforms, user_arch };

    if (
      (choice === "image" && !props.usingDefaultDataset) ||
      choice === "objectdetection"
    ) {
      const formData = new FormData();
      formData.append("file", uploadFile);
      await uploadToBackend(formData);
    }
    const trainResult = await train_and_output(
      choice,
      functionMap[choice][1](paramList)
    );
    setResult(trainResult);
  };

  useEffect(() => {
    if (uploaded && trainParams) {
      train_and_output(
        trainParams.choice,
        functionMap[trainParams.choice][1](trainParams.paramList)
      );
      setUploaded(false);
      setTrainParams(null);
    }
  }, [uploaded, trainParams]);

  useEffect(() => {
    if (result) {
      if (result.success) {
        if (props.email?.length) {
          sendEmail(props.email, props.problemType);
        }
        toast.success(choice === "objectdetection" ? "Detection successful! Scroll to see results!" : "Training successful! Scroll to see results!");
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
          backgroundColor: pendingResponse ? COLORS.disabled : null,
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

TrainButton.propTypes = {
  addedLayers: PropTypes.array,
  email: PropTypes.string,
  trainTransforms: PropTypes.array,
  testTransforms: PropTypes.array,
  transforms: PropTypes.array,
  setDLPBackendResponse: PropTypes.func.isRequired,
  choice: PropTypes.string,
  style: PropTypes.object,
  problemType: PropTypes.string,
  usingDefaultDataset: PropTypes.string,
  uploadFile: PropTypes.object,
  customModelName: PropTypes.string,
};

export default TrainButton;
