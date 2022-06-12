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
  ref2: PropTypes.oneOfType([
    PropTypes.func,
    PropTypes.shape({ current: PropTypes.instanceOf(Element) }),
  ]),
  dataTestid: PropTypes.node,
};

export default RectContainer;

const styles = {
  container: {
    width: 110,
    height: 60,
    padding: 10,
    margin: 7.5,
    border: `5px solid transparent`,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
  },
};
