import React from "react";
import PropTypes from "prop-types";
import { COLORS } from "../constants";

const BackgroundLayout = (props) => {
  return <div style={styles.layoutBackground}>{props.children}</div>;
};

BackgroundLayout.propTypes = {
  children: PropTypes.node,
};

export default BackgroundLayout;

const styles = {
  layoutBackground: {
    backgroundColor: COLORS.background,
    padding: 10,
    marginVertical: 10,
    display: "flex",
    flexDirection: "row",
    flexWrap: "wrap",
  },
};
