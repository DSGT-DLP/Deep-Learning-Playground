import React, { useState } from "react";
import Button from "react-bootstrap/Button";
import Form from "react-bootstrap/Form";

// import PropTypes from "prop-types";
// import { useNavigate } from "react-router-dom";
import { useSelector } from "react-redux";

const SettingsBlock = () => {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const signedInUserEmail = useSelector((state) => state.currentUser.email);
  const signedInUserName = useSelector(
    (state) => state.currentUser.displayName
  );

  const handleUpdateUser = () => {
    console.log(fullName + email + password);
  };
  return (
    <Form>
      <h2>View or Change your Account Settings </h2>
      <Form.Group className="mb-3" controlId="update-name">
        <Form.Label>Full Name</Form.Label>
        <Form.Control
          placeholder={signedInUserName}
          onBlur={(e) => setFullName(e.target.value)}
          size="lg"
        />
      </Form.Group>
      <Form.Group className="mb-3" controlId="update-email">
        <Form.Label>Email address</Form.Label>
        <Form.Control
          type="email"
          placeholder={signedInUserEmail}
          onBlur={(e) => setEmail(e.target.value)}
          autoComplete="email"
          size="lg"
        />
      </Form.Group>
      <Form.Group className="mb-3" controlId="update-password">
        <Form.Label>Password</Form.Label>
        <Form.Control
          type="password"
          placeholder={"New Password"}
          onBlur={(e) => setPassword(e.target.value)}
          aria-describedby="passwordHelpBlock"
          size="lg"
        />
        <Form.Text id="passwordHelpBlock" muted>
          Your password must be 8-20 characters long, contain letters and
          numbers, and must not contain spaces, special characters, or emoji.
        </Form.Text>
      </Form.Group>
      <Form.Group className="mb-3" controlId="update-check-password">
        <Form.Label>Re-Type Password</Form.Label>
        <Form.Control
          type="password"
          placeholder={"New Password"}
          onBlur={(e) => setPassword(e.target.value)}
          size="lg"
        />
      </Form.Group>
      <div className="email-buttons d-flex flex-column" onClick={handleUpdateUser}>
        <Button id="log-in" className="mb-2">
          Update Profile
        </Button>
      </div>
    </Form>
  );
};

const AccountSettings = () => {
  // const signedInUserEmail = useSelector((state) => state.currentUser.email);
  // const navigate = useNavigate();
  return (
    <div id="accountSettings">
      <div id="header-section">
        <h1 className="headers">User Settings</h1>
      </div>
      <div className="sections" id="User Settigs">
        <SettingsBlock />
      </div>
    </div>
  );
};

export default AccountSettings;
