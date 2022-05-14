import React, { useState } from "react";
import PropTypes from "prop-types";
import RectContainer from "./RectContainer";
import { COLORS, GENERAL_STYLES, LAYOUT } from "../constants";

const InputOutputPromptResponse = (props) => {
  const { isInput, value, setValue } = props;
  const prompt = isInput ? "Input Size:" : "Output size:";
  return (
    <div style={{ ...LAYOUT.row, alignItems: "center" }}>
      <p style={styles.input_prompt}>{prompt}</p>
      <input
        type="number"
        min={0}
        max={1000}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        style={styles.input_text}
      />
    </div>
  );
};

const AddedLayer = (props) => {
  const [inputSize, setInputSize] = useState(5);
  const [outputSize, setOutputSize] = useState(5);

  return (
    <div style={LAYOUT.column}>
      <RectContainer style={{ backgroundColor: COLORS.layer }}>
        <button style={styles.delete_btn} onClick={props.onDelete}>
          ❌
        </button>
        <p style={styles.text}>{props.text}</p>
      </RectContainer>
      <div style={styles.input_box}>
        <InputOutputPromptResponse
          isInput
          value={inputSize}
          setValue={setInputSize}
        />
        <InputOutputPromptResponse
          value={outputSize}
          setValue={setOutputSize}
        />
      </div>
    </div>
  );
};

AddedLayer.propTypes = {
  text: PropTypes.string.isRequired,
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
  text: { ...GENERAL_STYLES.p, color: COLORS.white, fontSize: 25 },
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
