import React from "react";
import PropTypes from "prop-types";
import { COLORS } from "../constants";

const RectContainer = (props) => {
  return (
    <div style={{ ...styles.container, ...props.style }}>{props.children}</div>
  );
};

RectContainer.propTypes = {
  children: PropTypes.node,
  style: PropTypes.object,
};

export default RectContainer;

const styles = {
  container: {
    width: 100,
    height: 50,
    padding: 20,
    marginRight: 10,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
};
