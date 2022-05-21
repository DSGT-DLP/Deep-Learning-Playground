import React from "react";
import PropTypes from "prop-types";
import RectContainer from "./RectContainer";
import { COLORS, GENERAL_STYLES } from "../constants";

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
  } = props;

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

  const onClick = () => {
    const user_arch = make_user_arch();
    console.log(
      user_arch,
      criterion,
      optimizerName,
      problemType,
      usingDefaultDataset,
      epochs
    );
  };

  return (
    <RectContainer style={styles.container}>
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
};

export default TrainButton;

const styles = {
  container: {
    backgroundColor: COLORS.dark_blue,
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
    fontSize: 20,
    color: "white",
  },
};