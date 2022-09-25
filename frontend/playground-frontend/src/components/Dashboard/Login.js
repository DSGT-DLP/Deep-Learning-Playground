import React, { useState } from "react";
import Button from "react-bootstrap/Button";
import Form from "react-bootstrap/Form";
import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signInWithRedirect,
} from "firebase/auth";
import { auth, googleProvider, githubProvider } from "../../firebase";
import { toast } from "react-toastify";
import { setCurrentUser } from "../../redux/userLogin";
import { useDispatch } from "react-redux";
import GoogleLogo from "../../images/logos/google.png";
import GithubLogo from "../../images/logos/github.png";
import { useNavigate } from "react-router-dom";

const Login = () => {
  const [isRegistering, setIsRegistering] = useState(false);
  const [email, setEmail] = useState();
  const [password, setPassword] = useState();
  const dispatch = useDispatch();
  const navigate = useNavigate();

  const updateCurrentUser = async (userCredential) => {
    const user = userCredential.user;
    const userData = {
      email: user.email,
      uid: user.uid,
      displayName: user.displayName,
      emailVerified: user.emailVerified,
    };
    await dispatch(setCurrentUser(userData));
    navigate("/dashboard");
  };

  const signInWithPassword = () => {
    signInWithEmailAndPassword(auth, email, password)
      .then((userCredential) => {
        updateCurrentUser(userCredential);
        toast.success(`Signed in with email ${userCredential.user.email}`, {
          autoClose: 1000,
        });
      })
      .catch((error) => toast.error(`Error: ${error.code}`));
  };

  const registerWithPassword = () => {
    createUserWithEmailAndPassword(auth, email, password)
      .then((userCredential) => {
        updateCurrentUser(userCredential);
        toast.success(`Registered with email ${userCredential.user.email}`, {
          autoClose: 1000,
        });
      })
      .catch((error) => toast.error(`Error: ${error.code}`));
  };

  const handleSignInRegister = () => {
    if (isRegistering) registerWithPassword();
    else signInWithPassword();
  };

  const Title = (
    <>
      <h1 className="title mb-5">
        No-code Solution for <br />
        Machine Learning
      </h1>
      <p className="description text-center mb-4">
        DLP is a playground where you can experiment with machine learning tools
        by inputting a dataset and use PyTorch modules without writing any code
      </p>
    </>
  );

  const SocialLogins = (
    <>
      <div className="d-flex justify-content-evenly mb-5">
        <Button
          className="login-button google"
          onClick={() => signInWithRedirect(auth, googleProvider)}
        >
          <img src={GoogleLogo} />
        </Button>
        <Button
          className="login-button github"
          onClick={() => {
            signInWithRedirect(auth, githubProvider);
          }}
        >
          <img src={GithubLogo} />
        </Button>
      </div>
    </>
  );

  const EmailPasswordInput = (
    <>
      {isRegistering && (
        <Form.Group className="mb-3" controlId="login-name">
          <Form.Label>Name</Form.Label>
          <Form.Control placeholder="Enter name" />
        </Form.Group>
      )}

      <Form.Group className="mb-3" controlId="login-email">
        <Form.Label>Email address</Form.Label>
        <Form.Control
          type="email"
          placeholder="someone@example.com"
          onBlur={(e) => setEmail(e.target.value)}
        />
      </Form.Group>

      <Form.Group className="mb-5" controlId="login-password">
        <Form.Label>Password</Form.Label>
        <Form.Control
          type="password"
          placeholder="Password"
          onBlur={(e) => setPassword(e.target.value)}
        />
      </Form.Group>
      <div className="email-buttons d-flex flex-column">
        <Button id="log-in" className="mb-2" onClick={handleSignInRegister}>
          {isRegistering ? "Register" : "Log in"}
        </Button>
        <a href="#" id="sign-up" onClick={() => setIsRegistering((e) => !e)}>
          {isRegistering ? "Log in" : "Register"}
        </a>
      </div>
    </>
  );

  return (
    <div id="login-page" className="text-center">
      <div className="main-container mt-5 mb-5">
        {Title}

        <Form className="form-container p-5">
          {SocialLogins}
          {EmailPasswordInput}
        </Form>
      </div>
    </div>
  );
};

export default Login;
