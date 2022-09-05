import React from "react";
import PropTypes from "prop-types";

const DButton = (props) => {
  const { text, onClick, style, disabled, className } = props;
  return (
    <button
      className={className || "btn btn-primary"}
      onClick={onClick}
      disabled={disabled}
      style={style}
    >
      {text}
    </button>
  );
};

DButton.propTypes = {};

export default DButton;
