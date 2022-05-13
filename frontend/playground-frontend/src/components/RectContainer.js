import React from "react";
import PropTypes from "prop-types";
import { COLORS } from "../constants";

const RectContainer = (props) => {
  return (
    <div
      style={{ ...styles.container, ...props.style }}
      ref={props.ref2}
      data-testid={props.dataTestid}
    >
      {props.children}
    </div>
  );
};

RectContainer.propTypes = {
  children: PropTypes.node,
  style: PropTypes.object,
  ref2: PropTypes.node,
  dataTestid: PropTypes.node,
};

export default RectContainer;

const styles = {
  container: {
    width: 100,
    height: 50,
    padding: 20,
    marginRight: 10,
    border: `5px solid transparent`,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
};
