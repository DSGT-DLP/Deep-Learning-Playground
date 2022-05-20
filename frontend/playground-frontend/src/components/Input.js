import React from "react";
import PropTypes from "prop-types";
import RectContainer from "./RectContainer";
import { COLORS, GENERAL_STYLES, LAYOUT } from "../constants";
import { DropDown } from "../components";
import { CRITERIONS } from "../settings";

const Input = (props) => {
  const { queryText, options, onChange, defaultValue, freeInputCustomProps } =
    props;

  return (
    <div style={{ ...LAYOUT.row, marginRight: 15 }}>
      <div style={styles.queryContainer}>
        <p style={styles.queryText}>{queryText}</p>
      </div>
      <div style={styles.responseContainer}>
        {options ? (
          <DropDown
            options={options}
            onChange={onChange}
            defaultValue={defaultValue}
          />
        ) : (
          <input
            style={styles.inputText}
            placeholder="Type..."
            maxLength={64}
            {...freeInputCustomProps}
            defaultValue={defaultValue}
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
  freeInputCustomProps: PropTypes.object,
  defaultValue: PropTypes.oneOfType([
    PropTypes.object,
    PropTypes.number,
    PropTypes.string,
  ]),
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
