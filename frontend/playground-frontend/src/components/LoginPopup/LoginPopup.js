import React, { useEffect } from "react";
import StyledFirebaseAuth from "./StyledFirebaseAuth";
import firebase from "firebase/compat/app";
import "firebase/compat/auth";
import PropTypes from "prop-types";
import { app } from "../../firebase";

// Configure FirebaseUI.
const uiConfig = {
  // Popup signin flow rather than redirect flow.
  signInFlow: "popup",
  // Redirect to / after sign in is successful. Alternatively you can provide a callbacks.signInSuccess function.
  signInSuccessUrl: "/",
  // We will display Google and Facebook as auth providers.
  signInOptions: [
    firebase.auth.EmailAuthProvider.PROVIDER_ID,
    firebase.auth.GoogleAuthProvider.PROVIDER_ID,
  ],
};

const LoginPopup = ({ setShowLogin }) => {
  useEffect(() => {
    document.getElementById("app").style.overflow = "hidden";
  }, []);
  return (
    <>
      <div
        id="page-mask"
        onClick={() => {
          setShowLogin(false);
          document.getElementById("app").style.overflow = null;
        }}
      ></div>
      <div id="login-popup">
        <h1 id="login-popup-h1">Log in or Sign up</h1>
        <StyledFirebaseAuth
          uiConfig={uiConfig}
          firebaseAuth={app.auth()}
          className={"login-page"}
          uiCallback={null}
        />
      </div>
    </>
  );
};

export default LoginPopup;

LoginPopup.propTypes = {
  setShowLogin: PropTypes.func.isRequired,
};
