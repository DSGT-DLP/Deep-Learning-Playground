import React, { useState } from "react";
import Button from "react-bootstrap/Button";
import Form from "react-bootstrap/Form";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import {
  createUserWithEmailAndPassword,
  signInWithRedirect,
} from "firebase/auth";
import { auth, googleProvider } from "../../firebase";
import { getRedirectResult, GoogleAuthProvider } from "firebase/auth";
import { toast } from "react-toastify";
import { useSelector, useDispatch } from "react-redux";
import { setCurrentUser } from "../../redux/userLogin";

const Login = () => {
  const [isRegistering, setIsRegistering] = useState(false);
  const dispatch = useDispatch();
  const user = useSelector((state) => state.currentUser.email)

  const handleRegister = (email, password) => {
    createUserWithEmailAndPassword(auth, email, password)
      .then((userCredential) => {})
      .catch((error) => toast.error(error.code));
  };

  getRedirectResult(auth)
    .then((result) => {
      const user = result.user;
      console.log(user);
    })
    .catch((error) => {
      toast.error(error.code);
    });

  const handleGoogleSignIn = () => {
    signInWithRedirect(auth, googleProvider);
  };

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
          <Row>
            <Col>
              <div>dwas</div>
            </Col>
            <Col>
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
                <Button
                  id="log-in"
                  className="mb-2"
                  onClick={handleGoogleSignIn}
                >
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
            </Col>
          </Row>
        </Form>
      </div>
    </div>
  );
};

export default Login;
