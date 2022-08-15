import React, { useState, useEffect } from "react";
import PropTypes from "prop-types";
import RectContainer from "./RectContainer";
import { COLORS, GENERAL_STYLES } from "../../constants";
import {
  validateTabularInputs,
  sendTabularJSON,
  validateImageInputs,
  sendPretrainedJSON,
  validatePretrainedInput,
  sendImageJSON,
} from "../helper_functions/TrainButtonFunctions";
import {
  socket,
  sendEmail,
  train_and_output,
} from "../helper_functions/TalkWithBackend";
import { Circle } from "rc-progress";
import { toast } from "react-toastify";

const TrainButton = (props) => {
  const { setDLPBackendResponse, choice = "tabular", style } = props;

  const [pendingResponse, setPendingResponse] = useState(false);
  const [progress, setProgress] = useState(null);
  const [result, setResult] = useState(null);
  const [uploaded, setUploaded] = useState(false);
  const [trainParams, setTrainParams] = useState(null);

  useEffect(() => {
    socket.on("trainingProgress", (progressData) => {
      // triggered by send_progress() function
      setProgress(Number.parseFloat(progressData));
    });
    socket.on("trainingResult", (resultData) => {
      setResult(resultData);
    });
    socket.on("uploadComplete", () => {
      setUploaded(true);
    });
  }, [socket]);

  const reset = () => {
    setPendingResponse(false);
    setProgress(null);
    setResult(null);
  };

  styles = { ...styles, ...style }; // style would take precedence

  const make_obj_param_list = (obj_list) => {
    if (!obj_list) return; // ValidateInputs throw error in case of empty things. This is to prevent an unnecessary errors in case of creating a layer

    // making a array of relating methods (like "nn.Linear") with their parameters (in_feature, out_feature) by including all methods and their parameters to make something like:
    // ["nn.Linear(4, 10)", "nn.ReLU()", "nn.Linear(10, 3)", "nn.Softmax()"] OR
    // ["transforms.ToTensor()", "transforms.RandomHorizontalFlip(0.8)"]

    const user_arch = [];
    obj_list.forEach((obj_list_item) => {
      const parameters = obj_list_item.parameters;
      let parameter_call_input = "";
      const parameters_to_be_added = Array(Object.keys(parameters).length);
      Object.values(parameters).forEach((v) => {
        parameters_to_be_added[v.index] = v.value;
      });
      parameters_to_be_added.forEach((e) => {
        parameter_call_input += e + ",";
      });
      // removing the last ','
      parameter_call_input = parameter_call_input.slice(0, -1);

      const callback = `${obj_list_item.object_name}(${parameter_call_input})`;
      user_arch.push(callback);
    });
    return user_arch;
  };

  const functionMap = {
    tabular: [validateTabularInputs, sendTabularJSON],
    image: [validateImageInputs, sendImageJSON],
    pretrained: [validatePretrainedInput, sendPretrainedJSON],
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
    setDLPBackendResponse(undefined);
    setProgress(0);

    const user_arch = make_obj_param_list(props.addedLayers);
    let trainTransforms = 0;
    let testTransforms = 0;
    if (props.trainTransforms) {
      trainTransforms = make_obj_param_list(props.trainTransforms);
      testTransforms = make_obj_param_list(props.testTransforms);
    }

    if (!validateInputs(user_arch)) {
      setPendingResponse(false);
      setProgress(null);
      return;
    }

    const paramList = { ...props, trainTransforms, testTransforms, user_arch };

    if (choice === "image" && !props.usingDefaultDataset) {
      setTrainParams({ choice, paramList });
      document.getElementById("fileUploadInput")?.click();
    } else {
      train_and_output(choice, functionMap[choice][1](paramList));
    }
  };

  useEffect(() => {
    if (uploaded && trainParams) {
      train_and_output(trainParams.choice, functionMap[trainParams.choice][1](trainParams.paramList));
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
        toast.success("Training successful! Scroll to see results!");
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
      <RectContainer
        style={{
          ...styles.container,
          backgroundColor: pendingResponse ? COLORS.disabled : COLORS.dark_blue,
        }}
      >
        <button
          style={{
            ...styles.button,
            cursor: pendingResponse ? "wait" : "pointer",
          }}
          onClick={onClick}
          disabled={pendingResponse}
        >
          Train!
        </button>
      </RectContainer>
      {pendingResponse ? (
        <div style={{ marginLeft: 5, marginTop: 10, width: 90, height: 90 }}>
          <Circle percent={progress} strokeWidth={4} />
        </div>
      ) : null}
    </>
  );
};

TrainButton.propTypes = {
  addedLayers: PropTypes.array,
  email: PropTypes.string,
  trainTransforms: PropTypes.array,
  testTransforms: PropTypes.array,
  setDLPBackendResponse: PropTypes.func.isRequired,
  choice: PropTypes.string,
  style: PropTypes.object,
  problemType: PropTypes.string,
  usingDefaultDataset: PropTypes.string,
};

export default TrainButton;

let styles = {
  container: {
    padding: 0,
    width: 130,
    height: 80,
  },
  button: {
    backgroundColor: "transparent",
    border: "none",
    cursor: "pointer",
    height: "100%",
    width: "100%",
    ...GENERAL_STYLES.p,
    fontSize: 25,
    color: "white",
  },
};
