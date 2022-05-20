import React from "react";
import PropTypes from "prop-types";
import RectContainer from "./RectContainer";
import { COLORS, GENERAL_STYLES, LAYOUT } from "../constants";

const Input = (props) => {
  return (
    <div style={LAYOUT.row}>
      <div style={styles.queryContainer}>
        <p style={styles.queryText}>Criterion</p>
      </div>
      <div style={styles.responseContainer}>
        <button
          style={styles.responseDropDownButton}
          onClick={() => console.log(111)}
        >
          ▼
        </button>
        <p style={styles.responseText}>CELOSS</p>
      </div>
    </div>
  );
};

Input.propTypes = {};

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
    width: 130,
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
};
