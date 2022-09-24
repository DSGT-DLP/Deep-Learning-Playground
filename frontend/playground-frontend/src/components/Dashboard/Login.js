import React, { useState } from "react";
import Button from "react-bootstrap/Button";
import Form from "react-bootstrap/Form";

const Login = () => {
  const [isRegistering, setIsRegistering] = useState(false);
  return (
    <div id="login-page" className="text-center">
      <div className="main-container mt-5">
        <h1 className="title mb-5">
          No-code Solution for <br />
          Machine Learning
        </h1>
        <p className="description text-center mb-4">
          DLP is a playground where you can experiment with machine learning
          tools by inputting a dataset and use PyTorch modules without writing
          any code
        </p>

        <Form className="form-container p-5">
          {isRegistering && (
            <Form.Group className="mb-3" controlId="login-name">
              <Form.Label>Name</Form.Label>
              <Form.Control placeholder="Enter name" />
            </Form.Group>
          )}

          <Form.Group className="mb-3" controlId="login-email">
            <Form.Label>Email address</Form.Label>
            <Form.Control type="email" placeholder="someone@example.com" />
          </Form.Group>

          <Form.Group className="mb-3" controlId="login-password">
            <Form.Label>Password</Form.Label>
            <Form.Control type="password" placeholder="Password" />
          </Form.Group>
          <div className="buttons d-flex flex-column">
            <Button id="log-in" className="mb-2">
              {isRegistering ? "Register" : "Log in"}
            </Button>
            <a
              href="#"
              id="sign-up"
              onClick={() => setIsRegistering((e) => !e)}
            >
              {isRegistering ? "Log in" : "Register"}
            </a>
          </div>
        </Form>
      </div>
    </div>
  );
};

export default Login;
