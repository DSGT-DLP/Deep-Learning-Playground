import React from "react";
import PropTypes from "prop-types";

const RectContainer = (props) => {
  return (
    <div
      style={props.style}
      ref={props.ref2}
      className={props.className}
      data-testid={props.dataTestid}
    >
      {props.children}
    </div>
  );
};

RectContainer.propTypes = {
  children: PropTypes.node,
  style: PropTypes.object,
  className: PropTypes.string,
  ref2: PropTypes.oneOfType([
    PropTypes.func,
    PropTypes.shape({ current: PropTypes.instanceOf(Element) }),
  ]),
  dataTestid: PropTypes.node,
};

export default RectContainer;

