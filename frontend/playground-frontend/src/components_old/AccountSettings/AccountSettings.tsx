import React, { useState } from "react";
import Button from "react-bootstrap/Button";
import Form from "react-bootstrap/Form";
import { updateUserSettings } from "../../firebase";
import { useAppSelector } from "../../redux/hooks";

const SettingsBlock = () => {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [checkPassword, setCheckedPassword] = useState("");
  const user = useAppSelector((state) => state.currentUser.user);

  const handleUpdateUser = async () => {
    if (password !== checkPassword) {
      alert("Passwords don't Match");
    } else {
      await updateUserSettings(fullName, email, password);
    }
  };

  if (!user) {
    return <></>;
  }
  return (
    <Form>
      <h2>View or Change your Account Settings </h2>
      <Form.Group className="mb-3" controlId="update-name">
        <Form.Label>Full Name</Form.Label>
        <Form.Control
          placeholder={user.displayName}
          onBlur={(e) => setFullName(e.target.value)}
          size="lg"
        />
      </Form.Group>
      <Form.Group className="mb-3" controlId="update-email">
        <Form.Label>Email address</Form.Label>
        <Form.Control
          type="email"
          placeholder={user.email}
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
      </Form.Group>
      <Form.Group className="mb-3" controlId="update-check-password">
        <Form.Label>Re-Type Password</Form.Label>
        <Form.Control
          type="password"
          placeholder={"New Password"}
          onBlur={(e) => setCheckedPassword(e.target.value)}
          size="lg"
        />
      </Form.Group>
      <div
        className="email-buttons d-flex flex-column"
        onClick={handleUpdateUser}
      >
        <Button id="update-profile" className="mb-2">
          Update Profile
        </Button>
      </div>
    </Form>
  );
};

const AccountSettings = () => {
  return (
    <div id="accountSettings">
      <div id="header-section">
        <h1 className="headers">User Settings</h1>
      </div>
      <div className="sections" id="User Settings" data-testid="user-settings">
        <SettingsBlock />
      </div>
    </div>
  );
};

export default AccountSettings;