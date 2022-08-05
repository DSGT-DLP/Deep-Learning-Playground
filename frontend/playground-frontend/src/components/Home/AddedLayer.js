import React from "react";
import PropTypes from "prop-types";
import RectContainer from "./RectContainer";
import { COLORS, GENERAL_STYLES, LAYOUT } from "../../constants";

const _InputOutputPromptResponse = (props) => {
  const {
    param_key,
    allParamInputs,
    setAddedLayers,
    thisLayerIndex,
  } = props;
  const { parameter_name, value } = allParamInputs[param_key];

  return (
    <div style={{ ...LAYOUT.row, alignItems: "center" }}>
      <p style={styles.input_prompt}>{`${parameter_name}:`}</p>
      <input
        value={value}
        onChange={(e) =>
          // updates the addedLayers state with the current user input value of parameters
          setAddedLayers((currentAddedLayers) => {
            const copyCurrent = [...currentAddedLayers];
            const parameters = copyCurrent[thisLayerIndex].parameters;
            parameters[param_key].value = e.target.value;
            return copyCurrent;
          })
        }
        style={styles.input_text}
      />
    </div>
  );
};

const AddedLayer = (props) => {
  const { thisLayerIndex, addedLayers, setAddedLayers, onDelete, style } =
    props;
  const thisLayer = addedLayers[thisLayerIndex];
  const { display_name, parameters } = thisLayer;

  styles = {...styles, ...style}

  // converts the parameters object for each layer into an array of parameter objects
  const param_array = [];
  Object.keys(parameters).forEach((key) => {
    param_array.push(
      <_InputOutputPromptResponse
        key={key}
        param_key={key}
        allParamInputs={thisLayer.parameters}
        setAddedLayers={setAddedLayers}
        thisLayerIndex={thisLayerIndex}
      />
    );
  });

  return (
    <div style={LAYOUT.column}>
      <RectContainer style={styles.layer_box}>
        <button style={styles.delete_btn} onClick={onDelete}>
          ❌
        </button>
        <p style={styles.text}>{display_name}</p>
      </RectContainer>
      <div style={styles.input_box}>{param_array}</div>
    </div>
  );
};

_InputOutputPromptResponse.propTypes = {
  param_key: PropTypes.string.isRequired,
  allParamInputs: PropTypes.shape({
    parameter_name: PropTypes.string,
    value: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
  }).isRequired,
  setAddedLayers: PropTypes.func.isRequired,
  thisLayerIndex: PropTypes.number.isRequired,
  finalStyle: PropTypes.any,
};

AddedLayer.propTypes = {
  thisLayerIndex: PropTypes.number.isRequired,
  addedLayers: PropTypes.arrayOf(PropTypes.object).isRequired,
  setAddedLayers: PropTypes.func.isRequired,
  onDelete: PropTypes.func.isRequired,
  style: PropTypes.any,
};

export default AddedLayer;

let styles = {
  layer_box: {
    backgroundColor: COLORS.layer,
    width: 130,
  },
  delete_btn: {
    position: "absolute",
    top: 0,
    right: 0,
    backgroundColor: "transparent",
    borderWidth: 0,
  },
  text: { ...GENERAL_STYLES.p, color: "white", fontSize: 25 },
  input_box: {
    margin: 7.5,
    backgroundColor: "white",
    width: 150,
    paddingInline: 5,
  },
  input_prompt: {
    fontFamily: "Arial, Helvetica, sans-serif",
    fontSize: 15,
    fontWeight: "bold",
  },
  input_text: {
    borderWidth: 0.5,
    borderColor: COLORS.layer,
    borderRadius: 10,
    fontSize: 15,
    maxWidth: "45%",
    padding: 5,
  },
};
