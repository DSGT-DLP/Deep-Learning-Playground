import React from "react";
import PropTypes from "prop-types";
import RectContainer from "./RectContainer";
import { COLORS, GENERAL_STYLES } from "../constants";

const AddedLayer = (props) => {
  return (
    <RectContainer style={{ backgroundColor: COLORS.layer }}>
      <button style={styles.delete_btn} onClick={props.onDelete}>
        ❌
      </button>
      <p style={styles.text}>{props.text}</p>
    </RectContainer>
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
};
