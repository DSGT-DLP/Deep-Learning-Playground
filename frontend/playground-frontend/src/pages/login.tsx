import React, { useState, useEffect } from "react";
import Button from "react-bootstrap/Button";
import Form from "react-bootstrap/Form";
/*
import {
  signInWithPassword,
  registerWithPassword,
  signInWithGoogle,
  signInWithGithub,
  getRedirectResultFromFirebase,
} from "../../firebase";*/
/* import { setCurrentUser } from "../../redux/userLogin"; */
import GoogleLogo from "/public/images/logos/google.png";
import GithubLogo from "/public/images/logos/github.png";
import ReCAPTCHA from "react-google-recaptcha";
import { toast } from "react-toastify";
import { useAppDispatch, useAppSelector } from "@/common/redux/hooks";
import { User } from "firebase/auth";
import Image from "next/image";
import {
  signInViaGithubRedirect,
  signInViaGoogleRedirect,
} from "@/common/redux/userLogin";
import { useRouter } from "next/router";
import NavbarMain from "@/common/components/NavBarMain";

const Login = () => {
  const [isRegistering, setIsRegistering] = useState(false);
  const [fullName, setFullName] = useState<string>("");
  const [email, setEmail] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [recaptcha, setRecaptcha] = useState<string | null>(null);
  const dispatch = useAppDispatch();
  const user = useAppSelector((state) => state.currentUser.user);
  const router = useRouter();
  useEffect(() => {
    console.log(user);
  });

  const handleSignInRegister = async () => {
    //let newUser: User | null = null;
    if (isRegistering) {
      if (!recaptcha) {
        toast.error("Please complete recaptcha");
      } else if (fullName === "") {
        toast.error("Please enter a name");
      } else if (email === "") {
        toast.error("Please enter an email");
      } else if (password === "") {
        toast.error("Please enter a password");
      } else {
        //newUser = await registerWithPassword(email, password, fullName);
      }
    } else {
      //newUser = await signInWithPassword(email, password);
    } /*
    if (!newUser || !newUser.email || !newUser.displayName) return;
    const userData = {
      email: newUser.email,
      uid: newUser.uid,
      displayName: newUser.displayName,
      emailVerified: newUser.emailVerified,
    };
    dispatch(setCurrentUser(userData));

    navigate("/dashboard");*/
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
          style={{
            position: "relative",
          }}
          onClick={() => {
            //dispatch(signInViaGoogleRedirect());
            router.push("/about");
          }}
        >
          <Image
            src={GoogleLogo}
            alt={"Sign In With Google"}
            fill={true}
            style={{ objectFit: "contain", margin: "auto" }}
          />
        </Button>
        <Button
          className="login-button github"
          style={{ position: "relative" }}
          onClick={() => {
            dispatch(signInViaGithubRedirect());
          }}
        >
          <Image
            src={GithubLogo}
            alt={"Sign In With Github"}
            fill={true}
            style={{ objectFit: "contain", margin: "auto" }}
          />
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
            {/* <Link to="/forgot">Forgot Password?</Link> */}
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

      {isRegistering && process.env.REACT_APP_CAPTCHA_SITE_KEY && (
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
    <NavbarMain>
      <div
        id="login-page"
        className="text-center d-flex justify-content-center"
      >
        <div className="main-container mt-5 mb-5">
          {Title}

          <Form className="form-container p-5">
            {SocialLogins}
            {EmailPasswordInput}
          </Form>
        </div>
      </div>
    </NavbarMain>
  );
};

export default Login;
