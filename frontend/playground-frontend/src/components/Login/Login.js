import React from 'react';
import StyledFirebaseAuth from 'react-firebaseui/StyledFirebaseAuth';
import firebase from 'firebase/compat/app';
import 'firebase/compat/auth';
import { useNavigate } from "react-router-dom";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyCMq0RmoCB1G6JHwwAvVXdcaykAhvwfv4Q",
  authDomain: "dsgt-authentication.firebaseapp.com",
  projectId: "dsgt-authentication",
  storageBucket: "dsgt-authentication.appspot.com",
  messagingSenderId: "15938712141",
  appId: "1:15938712141:web:0ecd946393c0584e72a204",
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);

// Configure FirebaseUI.
const uiConfig = {
  // Popup signin flow rather than redirect flow.
  signInFlow: "popup",
  // Redirect to /signedIn after sign in is successful. Alternatively you can provide a callbacks.signInSuccess function.
  signInSuccessUrl: "/signedIn",
  // We will display Google and Facebook as auth providers.
  signInOptions: [
    firebase.auth.EmailAuthProvider.PROVIDER_ID,
    firebase.auth.GoogleAuthProvider.PROVIDER_ID,
  ],
};

const Info = () => (
  <div id="login-info">
    <p>Information goes here</p>
  </div>
);

const Login = () => {
  const navigate = useNavigate();
  return (
    <div id="login-section">
      <StyledFirebaseAuth uiConfig={uiConfig} firebaseAuth={firebase.auth()} />
    </div>
  );
};

const About = () => {
  return (
    <div id="login-page">
      <div id="login-main-section">
        <Info />
        <Login />
      </div>
      {/* <footer/> */}
    </div>
  );
};

export default About;
