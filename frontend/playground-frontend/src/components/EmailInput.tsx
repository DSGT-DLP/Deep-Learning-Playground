import React, { useState } from "react";
import { GENERAL_STYLES } from "../constants";
import { Form, SSRProvider } from "react-bootstrap";

interface EmailInputProps {
  email?: string;
  setEmail: React.Dispatch<React.SetStateAction<string>>;
}

const EmailInput = (props: EmailInputProps) => {
  const { email, setEmail } = props;
  const [emailNotValid, setValue] = useState(true);
  const validRegex = /^\S+@\S+\.\S+$/;

  function updateEmailInput(emailInput: string) {
    setEmail(emailInput);
  }

  function validateEmail(email: string) {
    if (!email?.length || !email.match(validRegex)) {
      setValue(true);
      return;
    }
    setValue(false);
  }

  return (
    <>
      <Form.Control
        style={{ width: "25%" }}
        maxLength={255}
        placeholder="someone@example.com"
        onChange={(e) => updateEmailInput(e.target.value)}
        onBlur={(e) => validateEmail(e.target.value)}
      />
      {email && emailNotValid && (
        <p style={GENERAL_STYLES.error_text}>Please enter a valid email</p>
      )}
    </>
  );
};

export default EmailInput;
