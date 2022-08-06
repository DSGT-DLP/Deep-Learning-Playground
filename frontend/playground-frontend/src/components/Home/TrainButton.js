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
import { socket, sendEmail ,train_and_output } from "../helper_functions/TalkWithBackend";
import { Circle } from 'rc-progress';
import { toast } from "react-toastify";

const TrainButton = (props) => {
  const {
    setDLPBackendResponse,
    choice = "tabular",
    style,
  } = props;

  const [pendingResponse, setPendingResponse] = useState(false);
  const [progress, setProgress] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(() => {
    socket.on("trainingProgress", (progressData) => {
      setProgress(Number.parseFloat(progressData));
    });
    socket.on("trainingResult", (resultData) => {
      setResult(resultData);
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

  const validateInputs = (user_arch) => {
    let alertMessage = "";
    if (choice === "tabular")
      alertMessage = validateTabularInputs(user_arch, props);
    if (choice === "image")
      alertMessage = validateImageInputs(user_arch, props);
    if (choice === "pretrained")
      alertMessage = validatePretrainedInput(user_arch, props);

    if (alertMessage.length === 0) return true;
    toast.error(alertMessage);
    return false;
  };

  const onClick = async () => {
    setPendingResponse(true);
    setDLPBackendResponse(undefined);
    setProgress(0);

    const user_arch = make_obj_param_list(props.addedLayers);

    if (!validateInputs(user_arch)) {
      setPendingResponse(false);
      setProgress(null);
      return;
    }

    if (choice === "tabular")
      train_and_output(
        choice,
        sendTabularJSON(user_arch, props)
      );
    if (choice === "image")
      train_and_output(
        choice,
        sendImageJSON(
          user_arch,
          make_obj_param_list(props.trainTransforms),
          make_obj_param_list(props.testTransforms),
          props
        )
      );
    if (choice === "pretrained")
      train_and_output(
        choice,
        sendPretrainedJSON(user_arch, props)
      );
  };

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
  addedLayers: PropTypes.any,
  email: PropTypes.string,
  trainTransforms: PropTypes.any,
  testTransforms: PropTypes.any,
  setDLPBackendResponse: PropTypes.any,
  choice: PropTypes.string,
  style: PropTypes.any,
  problemType: PropTypes.any,
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
