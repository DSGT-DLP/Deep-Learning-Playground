import React from "react";
import PropTypes from "prop-types";
import { COLORS, GENERAL_STYLES, LAYOUT } from "../../constants";

const _InputOutputPromptResponse = (props) => {
  const { param_key, allParamInputs, setAddedLayers, thisLayerIndex } = props;
  const { parameter_name, value, min, max } = allParamInputs[param_key];

  return (
    <div style={{ ...LAYOUT.row, alignItems: "center" }}>
      <p style={styles.input_prompt}>{`${parameter_name}:`}</p>&nbsp;
      <input
        type={parameter_name === "(H, W)" ? "" : "number"}
        min={min}
        max={max}
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
  const { thisLayerIndex, addedLayers, setAddedLayers, onDelete } = props;
  const thisLayer = addedLayers[thisLayerIndex];
  const { display_name, parameters } = thisLayer;

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
    <div className="layer-input">
      <div className="layer-box layer-container text-center d-flex justify-content-center align-items-center">
        <button className="delete-layer" onClick={onDelete}>
          ❌
        </button>
        {display_name}
      </div>
      {param_array.length ? (
        <div className="input-box">{param_array}</div>
      ) : null}
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
};

AddedLayer.propTypes = {
  thisLayerIndex: PropTypes.number.isRequired,
  addedLayers: PropTypes.arrayOf(PropTypes.object).isRequired,
  setAddedLayers: PropTypes.func.isRequired,
  onDelete: PropTypes.func.isRequired,
};

export default AddedLayer;

let styles = {
  layer_box: {
    backgroundColor: COLORS.layer,
    width: 130,
  },
  text: { ...GENERAL_STYLES.p, color: "white", fontSize: 25 },
  input_prompt: {
    fontSize: 15,
    fontWeight: "bold",
  },
  input_text: {
    borderWidth: 0.5,
    borderColor: COLORS.layer,
    borderRadius: "0.375rem",
    fontSize: 15,
    maxWidth: "45%",
    padding: 5,
  },
};
