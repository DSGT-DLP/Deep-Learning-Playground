import React, { useState, useEffect } from "react";
import PropTypes from "prop-types";
import RectContainer from "./RectContainer";
import { COLORS, GENERAL_STYLES, LAYOUT } from "../constants";

const InputOutputPromptResponse = (props) => {
  const { name, allParamInputs, setAllParamInputs } = props;
  return (
    <div style={{ ...LAYOUT.row, alignItems: "center" }}>
      <p style={styles.input_prompt}>{`${name}:`}</p>
      <input
        value={allParamInputs[name]}
        onChange={(e) =>
          setAllParamInputs((currentValue) => {
            currentValue[name] = e.target.value;
            return currentValue;
          })
        }
        style={styles.input_text}
      />
    </div>
  );
};

const AddedLayer = (props) => {
  const { display_name, parameters } = props.layer;

  // *Example*
  // "Input Size": undefined, "Output Size": 5,
  const [allParameterInputs, setAllParameterInputs] = useState({});

  return (
    <div style={LAYOUT.column}>
      <RectContainer style={{ backgroundColor: COLORS.layer }}>
        <button style={styles.delete_btn} onClick={props.onDelete}>
          ❌
        </button>
        <p style={styles.text}>{display_name}</p>
      </RectContainer>
      <div style={styles.input_box}>
        {parameters?.map((e) => (
          <InputOutputPromptResponse
            key={e.display_name}
            name={e.display_name}
            allParamInputs={allParameterInputs}
            setAllParamInputs={setAllParameterInputs}
          />
        ))}
      </div>
    </div>
  );
};

AddedLayer.propTypes = {
  layer: PropTypes.object.isRequired,
  onDelete: PropTypes.func,
};

export default AddedLayer;

const styles = {
  delete_btn: {
    position: "absolute",
    top: 0,
    right: 0,
    backgroundColor: "transparent",
    borderWidth: 0,
  },
  text: { ...GENERAL_STYLES.p, color: "white", fontSize: 25 },
  input_box: {
    marginTop: 10,
    backgroundColor: "white",
    width: 130,
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
