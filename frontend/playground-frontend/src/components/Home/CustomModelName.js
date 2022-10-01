import React from "react";
import PropTypes from "prop-types";
import { TitleText } from "../index";
import { Form } from "react-bootstrap";

const CustomModelName = (props) => {
  const { customModelName, setCustomModelName } = props;
  return (
    <>
      <TitleText text="Custom Model Name" />
      <Form.Control
        className="model-name-input"
        placeholder="Give a model name"
        defaultValue={customModelName}
        onBlur={(e) => setCustomModelName(e.target.value)}
        maxLength={255}
      />
    </>
  );
};

CustomModelName.propTypes = {
  customModelName: PropTypes.string.isRequired,
  setCustomModelName: PropTypes.func.isRequired,
};

export default CustomModelName;
