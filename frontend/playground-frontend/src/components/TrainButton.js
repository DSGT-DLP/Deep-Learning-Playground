import React, { useState } from "react";
import PropTypes from "prop-types";
import RectContainer from "./RectContainer";
import { COLORS, GENERAL_STYLES } from "../constants";
import { train_and_output } from "../helper_functions/TalkWithBackend";

const TrainButton = (props) => {
  const {
    addedLayers,
    targetCol = null,
    features = null,
    problemType,
    criterion,
    optimizerName,
    usingDefaultDataset = null,
    shuffle,
    epochs,
    testSize,
    setDLPBackendResponse,
    csvData = null,
    fileURL = null,
    email
  } = props;

  const [backgroundColor, setBackgroundColor] = useState(COLORS.dark_blue);

  const make_user_arch = () => {
    // making a user_arch array by including all added layers and their parameters to make something like:
    // ["nn.Linear(4, 10)", "nn.ReLU()", "nn.Linear(10, 3)", "nn.Softmax()"]
    const user_arch = [];
    addedLayers.forEach((addedLayer) => {
      const parameters = addedLayer.parameters;
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

      const callback = `${addedLayer.object_name}(${parameter_call_input})`;
      user_arch.push(callback);
    });
    return user_arch;
  };

  const validateInputs = (user_arch) => {
    let alertMessage = "";
    if (!user_arch?.length)
      alertMessage += "At least one layer must be added. ";
    if (!criterion) alertMessage += "A criterion must be specified. ";
    if (!optimizerName) alertMessage += "An optimizer name must be specified. ";
    if (!problemType) alertMessage += "A problem type must be specified. ";
    if (!usingDefaultDataset) {
      if (!targetCol || !features?.length) {
        alertMessage +=
          "Must specify an input file, target, and features if not selecting default dataset. ";
      }
      for (let i = 0; i < features.length; i++) {
        if (targetCol === features[i]) {
          alertMessage +=
            "A column that is selected as the target column cannot also be a feature column. ";
          break;
        }
      }
      if (!csvData && !fileURL) {
        alertMessage +=
          "Must specify an input file either from local storage or from an internet URL. ";
      }
    }

    if (alertMessage.length === 0) return true;

    alert(alertMessage);
    return false;
  };

  const onClick = async () => {
    setBackgroundColor((currentColor) =>
      currentColor === COLORS.dark_blue ? COLORS.gold : COLORS.dark_blue
    );

    const user_arch = make_user_arch();
    if (!validateInputs(user_arch)) return;

    const csvDataStr = JSON.stringify(csvData);

    const response = await train_and_output(
      user_arch,
      criterion,
      optimizerName,
      problemType,
      targetCol,
      features,
      usingDefaultDataset,
      testSize,
      epochs,
      shuffle,
      csvDataStr,
      fileURL,
      email
    );

    setDLPBackendResponse(response);

    if (response.success === true) {
      alert("SUCCESS: Training successful! Scroll to see results!");
    } else if (response.message) {
      alert("FAILED: Training failed. Check output traceback message");
    } else {
      alert("FAILED: Training failed. Check your inputs");
    }
  };

  return (
    <RectContainer style={{ ...styles.container, backgroundColor }}>
      <button style={styles.button} onClick={onClick}>
        Train!
      </button>
    </RectContainer>
  );
};

TrainButton.propTypes = {
  addedLayers: PropTypes.arrayOf(PropTypes.object),
  targetCol: PropTypes.string,
  features: PropTypes.arrayOf(PropTypes.string),
  problemType: PropTypes.string,
  criterion: PropTypes.string,
  optimizerName: PropTypes.string,
  usingDefaultDataset: PropTypes.string,
  shuffle: PropTypes.bool,
  epochs: PropTypes.number,
  testSize: PropTypes.number,
};

export default TrainButton;

const styles = {
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
