import React from "react";
import PropTypes from "prop-types";

const DButton = (props) => {
  const { onClick, style, disabled, className, children } = props;
  return (
    <button
      className={className || "btn btn-primary"}
      onClick={onClick}
      disabled={disabled}
      style={style}
    >
      {props.children}
    </button>
  );
};

DButton.propTypes = {
  onClick: PropTypes.func,
  style: PropTypes.objectOf(PropTypes.string),
  disabled: PropTypes.bool,
  className: PropTypes.string,
  children: PropTypes.string,
};

export default DButton;
