import React, { useState } from "react";
import PropTypes from "prop-types";
import RectContainer from "./RectContainer";
import { COLORS, GENERAL_STYLES } from "../constants";
import { train_and_output } from "../TalkWithBackend";

const TrainButton = (props) => {
  const {
    addedLayers,
    targetCol,
    features,
    problemType,
    criterion,
    optimizerName,
    usingDefaultDataset,
    shuffle,
    epochs,
    testSize,
    set_dl_results_data,
    csvData = null,
    fileURL = null,
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
          "Must specify an input file, target, and features if not selecting default dataset.";
      }
      if (!csvData && !fileURL) {
        alertMessage +=
          "Must specify an input file either from local storage or from an internet URL.";
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

    const success = await train_and_output(
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
      set_dl_results_data,
      csvDataStr,
      fileURL
    );

    if (success === true) {
      alert("Training successful! Scroll to see results!");
    } else {
      alert("Training failed. Check inputs");
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
  usingDefaultDataset: PropTypes.bool,
  shuffle: PropTypes.bool,
  epochs: PropTypes.number,
  testSize: PropTypes.number,
  set_dl_results_data: PropTypes.func.isRequired,
};

export default TrainButton;

const styles = {
  container: {
    padding: 0,
    width: 130,
    height: 70,
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
