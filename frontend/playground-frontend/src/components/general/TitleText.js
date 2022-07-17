import React from "react";
import PropTypes from "prop-types";
import { COLORS, GENERAL_STYLES } from "../../constants";

const TitleText = (props) => {
  const { text } = props;
  return <p style={styles.titleText}>{text}</p>;
};

export default TitleText;

const styles = {
  titleText: { ...GENERAL_STYLES.p, color: COLORS.layer, fontSize: 20 },
};

TitleText.propTypes = {
  text: PropTypes.string.isRequired,
};
