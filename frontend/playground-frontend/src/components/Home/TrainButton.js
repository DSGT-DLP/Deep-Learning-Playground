import React, { useState } from "react";
import PropTypes from "prop-types";
import RectContainer from "./RectContainer";
import { COLORS, GENERAL_STYLES } from "../../constants";
import { train_and_output } from "../helper_functions/TalkWithBackend";
import {
  validateTabularInputs,
  sendTabularJSON,
  validateImageInputs,
  sendPretrainedJSON,
  validatePretrainedInput,
  sendImageJSON,
} from "../helper_functions/TrainButtonFunctions";

const TrainButton = (props) => {
  const {
    addedLayers,
    setDLPBackendResponse,
    csvDataInput = null,
    paramaters,
    choice = "tabular",
    style,
  } = props;

  const [pendingResponse, setPendingResponse] = useState(false);

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
      Object.entries(parameters).forEach((entry) => {
        const [k, v] = entry;
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
    alert(alertMessage);
    return false;
  };

  const onClick = async () => {
    setPendingResponse(true);
    setDLPBackendResponse(undefined);

    const user_arch = make_obj_param_list(props.addedLayers);

    if (!validateInputs(user_arch)) {
      setPendingResponse(false);
      return;
    }

    let response;
    if (choice === "tabular")
      response = await train_and_output(
        choice,
        sendTabularJSON(user_arch, props)
      );
    if (choice === "image") {
      response = await train_and_output(
        choice,
        sendImageJSON(
          user_arch,
          make_obj_param_list(props.trainTransforms),
          make_obj_param_list(props.testTransforms),
          props
        )
      );
    }
    if (choice === "pretrained")
      response = await train_and_output(
        choice,
        sendPretrainedJSON(user_arch, props)
      );

    setDLPBackendResponse(response);
    setPendingResponse(false);

    if (response.success === true) {
      alert("SUCCESS: Training successful! Scroll to see results!");
    } else if (response.message) {
      alert("FAILED: Training failed. Check output traceback message");
    } else {
      alert("FAILED: Training failed. Check your inputs");
    }
  };

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
        <div style={{ marginTop: 10 }}>
          <div className="loader" />
        </div>
      ) : null}
    </>
  );
};

// TrainButton.propTypes = {
//   addedLayers: PropTypes.arrayOf(PropTypes.object),
//   targetCol: PropTypes.string,
//   features: PropTypes.arrayOf(PropTypes.string),
//   problemType: PropTypes.string,
//   criterion: PropTypes.string,
//   optimizerName: PropTypes.string,
//   usingDefaultDataset: PropTypes.string,
//   shuffle: PropTypes.bool,
//   epochs: PropTypes.number,
//   testSize: PropTypes.number,
// };

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
