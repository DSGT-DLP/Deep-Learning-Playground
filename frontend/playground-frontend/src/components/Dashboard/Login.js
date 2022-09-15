import React from "react";
import { DButton } from "../index";
import DLP_logo from "../../images/logos/dlp_branding/dlp-logo.png";
import { LAYOUT } from "../../constants";

const Login = () => {
  return (
    <div id="login-page">
      <div className="logo">
        <img src={DLP_logo} style={{ height: "100%", aspectRatio: 1 }} />
        Deep Learning Playground
      </div>
      <div className="main-container">
        <h1>No-code Solution for Machine Learning</h1>
        <p>
          DLP is a playground where you can experiment with machine learning
          tools by inputting a dataset and use PyTorh modules without writing
          any code
        </p>

        <button></button>
      </div>
      <DButton>Log In</DButton>
      <a>Sign up</a>

      <div style={LAYOUT.row}>
        <DButton>About</DButton>
        <DButton>Wiki</DButton>
        <DButton>Donate</DButton>
      </div>
    </div>
  );
};

export default Login;
