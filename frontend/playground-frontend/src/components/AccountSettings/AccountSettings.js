import React, { useState } from "react";
import { Button, Form, Row, Col, Container } from "react-bootstrap";
import { updateUserSettings } from "../../firebase";
import { useSelector } from "react-redux";
import GoogleLogo from "../../images/logos/google.png";
import GithubLogo from "../../images/logos/github.png";
import { ImCross, ImCheckmark } from "react-icons/im";

const SettingsBlock = () => {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [checkPassword, setCheckedPassword] = useState("");
  const signedInUserEmail = useSelector((state) => state.currentUser.email);
  const signedInUserName = useSelector(
    (state) => state.currentUser.displayName
  );

  const handleUpdateUser = async () => {
    let user;
    if (password !== checkPassword) {
      alert("Passwords don't Match");
    } else {
      user = await updateUserSettings(fullName, email, password);
    }
    if (!user) return;
  };

  const UpdatePassword = (
    <>
      <h2>View or Change your Account Settings</h2>
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
          autoComplete="new-password"
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
          autoComplete="new-password"
          placeholder={"New Password"}
          onBlur={(e) => setCheckedPassword(e.target.value)}
          size="lg"
        />
      </Form.Group>
    </>
  );

  const LinkAccounts = () => {
    const accountsInfo = [
      {
        id: "Google",
        cssClass: "google",
        logo: GoogleLogo,
        status: "Unlinked",
      },
      {
        id: "GitHub",
        cssClass: "github",
        logo: GithubLogo,
        status: "Linked",
      },
    ];
    return (
      <>
        <h2>Link Accounts</h2>
        <Container className="mt-3 mb-5">
          {accountsInfo.map((account) => (
            <Row key={account.id} className="mt-2 mb-2">
              <Col md={3}>
                <Button className={`login-button ${account.cssClass}`}>
                  <img src={account.logo} />
                </Button>
              </Col>
              <Col className="d-flex align-items-center">
                {account.status === "Linked" ? (
                  <div className="d-flex align-items-center linked-acct">
                    <ImCheckmark className="me-2" />
                    Linked
                  </div>
                ) : (
                  <div className="d-flex align-items-center unlinked-acct">
                    <ImCross className="me-2" />
                    Unlinked
                  </div>
                )}
              </Col>
            </Row>
          ))}
        </Container>
      </>
    );
  };

  return (
    <Form>
      <LinkAccounts />
      <div className="email-buttons d-flex flex-column">
        {UpdatePassword}
        <Button id="update-profile" className="mb-2" onClick={handleUpdateUser}>
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
      <div className="sections" id="User Settigs">
        <SettingsBlock />
      </div>
    </div>
  );
};

export default AccountSettings;
