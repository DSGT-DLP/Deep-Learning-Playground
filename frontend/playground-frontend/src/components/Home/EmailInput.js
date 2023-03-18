import React, { useState } from "react";
import PropTypes from "prop-types";
import { GENERAL_STYLES } from "../../constants";
import { Form } from "react-bootstrap";

const EmailInput = (props) => {
  const { setEmail } = props;
  const [emailNotValid, setValue] = useState("");
  const validRegex = /^\S+@\S+\.\S+$/;

  function updateEmailInput(emailInput) {
    setEmail(emailInput);
  }

  function validateEmail(email) {
    if (!email?.length || !email.match(validRegex)) {
      setValue(true);
      return;
    }
    setValue(false);
  }

  return (
    <Form>
      <Form.Control
        style={{ width: "25%" }}
        maxLength={255}
        placeholder="someone@example.com"
        onChange={(e) => updateEmailInput(e.target.value)}
        onBlur={(e) => validateEmail(e.target.value)}
      />
      {emailNotValid && (
        <p style={GENERAL_STYLES.error_text}>Please enter a valid email</p>
      )}
    </Form>
  );
};

EmailInput.propTypes = {
  setEmail: PropTypes.func.isRequired,
};

export default EmailInput;
