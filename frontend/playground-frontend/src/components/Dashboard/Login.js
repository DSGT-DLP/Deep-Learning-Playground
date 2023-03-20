import { useState, useEffect } from "react";
import Button from "react-bootstrap/Button";
import Form from "react-bootstrap/Form";
import {
  signInWithPassword,
  registerWithPassword,
  signInWithGoogle,
  signInWithGithub,
} from "../../firebase";
import { setCurrentUser } from "../../redux/userLogin";
import { useDispatch } from "react-redux";
import GoogleLogo from "../../images/logos/google.png";
import GithubLogo from "../../images/logos/github.png";
import { useSelector } from "react-redux";
import { useNavigate, Link } from "react-router-dom";
import ReCAPTCHA from "react-google-recaptcha";
import { toast } from "react-toastify";

const Login = () => {
  const [isRegistering, setIsRegistering] = useState(false);
  const [fullName, setFullName] = useState();
  const [email, setEmail] = useState();
  const [password, setPassword] = useState();
  const [recaptcha, setRecaptcha] = useState("");
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const signedInUserEmail = useSelector((state) => state.currentUser.email);

  const handleSignInRegister = async () => {
    let user;
    if (isRegistering) {
      if (recaptcha !== "") {
        user = await registerWithPassword(email, password, fullName);
      } else {
        toast.error("Please complete recaptcha");
      }
    } else {
      user = await signInWithPassword(email, password);
    }

    if (!user) return;
    const userData = {
      email: user.email,
      uid: user.uid,
      displayName: user.displayName,
      emailVerified: user.emailVerified,
    };

    dispatch(setCurrentUser(userData));
    navigate("/dashboard");
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

  useEffect(() => {
    if (signedInUserEmail) navigate("/dashboard");
  }, [signedInUserEmail]);

  const SocialLogins = (
    <>
      <div className="d-flex justify-content-evenly mb-5">
        <Button
          className="login-button google"
          onClick={() => signInWithGoogle()}
        >
          <img src={GoogleLogo} />
        </Button>
        <Button
          className="login-button github"
          onClick={() => signInWithGithub()}
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
          <Form.Control
            placeholder="Enter name"
            onBlur={(e) => setFullName(e.target.value)}
            autoComplete="name"
          />
        </Form.Group>
      )}

      <Form.Group className="mb-3" controlId="login-email">
        <Form.Label>Email address</Form.Label>
        <Form.Control
          type="email"
          placeholder="someone@example.com"
          onBlur={(e) => setEmail(e.target.value)}
          autoComplete="email"
        />
      </Form.Group>

      <Form.Group className="mb-5" controlId="login-password">
        <Form.Label>Password</Form.Label>
        <Form.Control
          type="password"
          placeholder="Password"
          onBlur={(e) => setPassword(e.target.value)}
          autoComplete="current-password"
        />
        {!isRegistering && (
          <div className="link">
            <Link to="/forgot">Forgot Password?</Link>
          </div>
        )}
      </Form.Group>

      <div className="email-buttons d-flex flex-column">
        <Button id="log-in" className="mb-2" onClick={handleSignInRegister}>
          {isRegistering ? "Register" : "Log in"}
        </Button>
        <a href="#" id="sign-up" onClick={() => setIsRegistering((e) => !e)}>
          {isRegistering ? "Log in" : "Register"}
        </a>
      </div>

      {isRegistering && (
        <div className="reCaptcha">
          <ReCAPTCHA
            sitekey={process.env.REACT_APP_CAPTCHA_SITE_KEY}
            theme="dark"
            onChange={(e) => setRecaptcha(e)}
          />
        </div>
      )}
    </>
  );

  return (
    <div id="login-page" className="text-center d-flex justify-content-center">
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
