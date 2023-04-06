import React from "react";
import { Form } from "react-bootstrap";

interface CustomModelNamePropTypes {
  customModelName: string;
  setCustomModelName: React.Dispatch<React.SetStateAction<string>>;
}
const CustomModelName = (props: CustomModelNamePropTypes) => {
  const { customModelName, setCustomModelName } = props;
  return (
    <>
      <Form.Control
        className="model-name-input"
        placeholder="Give a custom model name"
        defaultValue={customModelName}
        onBlur={(e) => setCustomModelName(e.target.value)}
        maxLength={255}
      />
    </>
  );
};

export default CustomModelName;
