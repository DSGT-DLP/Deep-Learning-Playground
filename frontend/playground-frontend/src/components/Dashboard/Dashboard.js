import React from "react";
import PropTypes from "prop-types";
import { LAYOUT } from "../../constants";
import { DButton } from "../index";
import DSGTLogo from "../../images/logos/dlp_branding/dlp-logo.png";

const Dashboard = (props) => {
  return (
    <div style={{ ...LAYOUT.centerMiddle, ...LAYOUT.column, height: "100vh" }}>
      <img src={DSGTLogo} alt="DSGT Logo" width="300" aspectRatio="1" />
      <h1>Deep Learning Playground</h1>
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

Dashboard.propTypes = {};

export default Dashboard;
