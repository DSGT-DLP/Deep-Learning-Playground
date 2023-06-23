import React, { useState, useEffect } from "react";
import Button from "react-bootstrap/Button";
import Form from "react-bootstrap/Form";
import { auth } from "@/common/utils/firebase";
import { signInWithCustomToken } from "firebase/auth";
import { Magic } from "magic-sdk";
import GoogleLogo from "/public/images/logos/google.png";
import GithubLogo from "/public/images/logos/github.png";
import ReCAPTCHA from "react-google-recaptcha";
import { toast } from "react-toastify";
import { useAppDispatch, useAppSelector } from "@/common/redux/hooks";
import Image from "next/image";
import {
  isSignedIn,
  registerViaEmailAndPassword,
  signInViaEmailAndPassword,
  signInViaGithubRedirect,
  signInViaGoogleRedirect,
} from "@/common/redux/userLogin";
import NavbarMain from "@/common/components/NavBarMain";
import Link from "next/link";
import Footer from "@/common/components/Footer";
import { SerializedError } from "@reduxjs/toolkit";
import { useRouter } from "next/router";
import { httpsCallable } from "firebase/functions";

const Login = () => {
  const [isRegistering, setIsRegistering] = useState(false);
  const [fullName, setFullName] = useState<string>("");
  const [email, setEmail] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [recaptcha, setRecaptcha] = useState<string | null>(null);
  const user = useAppSelector((state) => state.currentUser.user);
  const router = useRouter();
  const dispatch = useAppDispatch();

  useEffect(() => {
    if (router.isReady && isSignedIn(user)) {
      router.replace({ pathname: "/dashboard" });
    }
  }, [user, router.isReady]);

  console.log(router.query);

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
          onClick={async () => {
            try {
              await dispatch(signInViaGoogleRedirect()).unwrap();
            } catch (e) {
              toast.error((e as SerializedError).message, {
                position: toast.POSITION.TOP_CENTER,
              });
            }
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
          onClick={async () => {
            try {
              await dispatch(signInViaGithubRedirect()).unwrap();
            } catch (e) {
              toast.error((e as SerializedError).message, {
                position: toast.POSITION.TOP_CENTER,
              });
            }
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
      <div className="d-flex justify-content-center mb-5">
        <Button
          variant="outline-light"
          className="login-button passwordless custom-button"
          onClick={async () => {
            try {
              const email = prompt("Enter your email address:");
              if (email) {
                const didToken = await magic.auth.loginWithEmailOTP({ email });

                // if didToken is not null, query the backend to find the UID of the user
                // **********
                // https://firebase.google.com/docs/auth/admin/manage-users
                // getAuth()
                // .getUserByEmail(email)
                // .then((userRecord) => {
                //   // See the UserRecord reference doc for the contents of userRecord.
                //   console.log(`Successfully fetched user data: ${userRecord.toJSON()}`);
                // })
                // .catch((error) => {
                //   console.log('Error fetching user data:', error);
                // });
                // **********
                // and sign in by changing the redux

                toast.success("Sign-in link sent to your email!", {
                  position: toast.POSITION.TOP_CENTER,
                });
              }
            } catch (e) {
              toast.error((e as { message: string }).message, {
                position: toast.POSITION.TOP_CENTER,
              });
            }
          }}
        >
          Sign in with Email Link
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
            <Link href="/forgot">Forgot Password?</Link>
          </div>
        )}
      </Form.Group>
      {isRegistering && process.env.REACT_APP_CAPTCHA_SITE_KEY && (
        <div className="reCaptcha">
          <ReCAPTCHA
            sitekey={process.env.REACT_APP_CAPTCHA_SITE_KEY}
            theme="dark"
            onChange={(e) => setRecaptcha(e)}
          />
        </div>
      )}
      <div className="email-buttons d-flex flex-column">
        <Button
          id="log-in"
          className="mb-2"
          onClick={async () => {
            if (isRegistering) {
              try {
                await dispatch(
                  registerViaEmailAndPassword({
                    email: email,
                    password: password,
                    displayName: fullName,
                    recaptcha: recaptcha,
                  })
                ).unwrap();
                toast.success(`Welcome ${fullName}`, {
                  position: toast.POSITION.TOP_CENTER,
                });
              } catch (e) {
                toast.error((e as SerializedError).message, {
                  position: toast.POSITION.TOP_CENTER,
                });
              }
            } else {
              try {
                await dispatch(
                  signInViaEmailAndPassword({ email, password })
                ).unwrap();
              } catch (e) {
                toast.error((e as SerializedError).message, {
                  position: toast.POSITION.TOP_CENTER,
                });
              }
            }
          }}
        >
          {isRegistering ? "Register" : "Log in"}
        </Button>
        <Button
          variant="outline-dark"
          id="sign-up"
          onClick={() => setIsRegistering((e) => !e)}
        >
          {isRegistering ? "Log in" : "Register"}
        </Button>
      </div>
    </>
  );

  if (user !== undefined) {
    return <></>;
  }

  return (
    <>
      <NavbarMain />
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
      <Footer />
    </>
  );
};

export default Login;

const magic = new Magic("pk_live_4572C928C88C9F0F", {
  network: "mainnet",
});
