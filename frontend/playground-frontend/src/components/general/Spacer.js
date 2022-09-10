import React from "react";
import PropTypes from "prop-types";

const Spacer = (props) => {
  return <div style={{ height: props.height || 1, width: props.width || 1 }} />;
};

Spacer.propTypes = {};

export default Spacer;
