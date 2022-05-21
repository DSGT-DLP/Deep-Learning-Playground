import React from "react";
import PropTypes from "prop-types";
import RectContainer from "./RectContainer";
import { COLORS, GENERAL_STYLES } from "../constants";
import { train_and_output } from "../TalkWithBackend";

const TrainButton = (props) => {
  return (
    <RectContainer style={styles.container}>
      <button style={styles.button} onClick={train_and_output}>
        Train!
      </button>
    </RectContainer>
  );
};

TrainButton.propTypes = {
  onClick: PropTypes.func,
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
