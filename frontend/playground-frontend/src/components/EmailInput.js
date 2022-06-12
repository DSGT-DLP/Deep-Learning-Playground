import React, { useState } from "react";
import TitleText from "./mini_components/TitleText";

const EmailInput = (props) => {
  const { setEmail } = props;
  const [emailNotValid, setValue] = useState("");
  var validRegex =
    /(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])/;

  function handleEmailInput(emailInput) {
    if (emailInput.length != 0) {
      if (emailInput.match(validRegex)) {
        setValue(false);
        setEmail(emailInput);
      } else {
        setValue(true);
        setEmail("");
      }
    } else {
      setValue(false);
    }
  }

  return (
    <>
      <TitleText text="Email" />
      <input
        style={{ width: "25%" }}
        placeholder="Optional email input"
        onChange={(e) => {
          handleEmailInput(e.target.value);
        }}
      />
      {emailNotValid && <p>Please enter a valid email</p>}
    </>
  );
};

export default EmailInput;
