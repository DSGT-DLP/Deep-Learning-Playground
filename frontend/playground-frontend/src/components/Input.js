import React from "react";
import PropTypes from "prop-types";
import RectContainer from "./RectContainer";
import { COLORS, GENERAL_STYLES, LAYOUT } from "../constants";
import { DropDown } from "../components";
import { CRITERIONS } from "../settings";

const Input = (props) => {
  return (
    <div style={{ ...LAYOUT.row, marginRight: 10 }}>
      <div style={styles.queryContainer}>
        <p style={styles.queryText}>{props.queryText}</p>
      </div>
      <div style={styles.responseContainer}>
        {props.options ? (
          <DropDown options={props.options} onChange={props.onChange} />
        ) : (
          <input
            style={styles.inputText}
            placeholder="10"
            maxLength={64}
            type={props.inputType}
          />
        )}
      </div>
    </div>
  );
};

Input.propTypes = {
  queryText: PropTypes.string.isRequired,
  options: PropTypes.arrayOf(PropTypes.object),
  onChange: PropTypes.func,
  inputType: PropTypes.string,
};

export default Input;

const styles = {
  queryContainer: {
    height: 50,
    width: 130,
    backgroundColor: COLORS.layer,
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  },
  queryText: {
    ...GENERAL_STYLES.p,
    color: "white",
    textAlign: "center",
    fontSize: 18,
  },
  responseContainer: {
    height: 50,
    width: 150,
    backgroundColor: COLORS.addLayer,
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  },
  responseText: {
    ...GENERAL_STYLES.p,
    color: "black",
    textAlign: "center",
    fontSize: 18,
  },
  responseDropDownButton: { border: "none", fontSize: 18, cursor: "pointer" },
  inputText: {
    ...GENERAL_STYLES.p,
    border: "none",
    backgroundColor: "transparent",
    width: "100%",
    textAlign: "center",
    fontSize: 18,
  },
};
