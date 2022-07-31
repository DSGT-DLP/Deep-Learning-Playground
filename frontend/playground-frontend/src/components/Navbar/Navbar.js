import DSGTLogo from "../../images/logos/dlp_branding/dlp-logo.png";
import React from "react";
import { Link } from "react-router-dom";

const Navbar = () => {
  return (
    <div className="header-footer" id="nav-bar">
      <a href="/" className="image-title">
        <img src={DSGTLogo} alt="DSGT Logo" width="60" height="60" />
        <div style={{ marginRight: 10 }} />
        Deep Learning Playground
      </a>
      <ul className="nav">
        <li id="title-name"></li>
        <li className="navElement">
          <Link to="/">Home</Link>
        </li>
        <li className="navElement">
          <Link to="/About">About</Link>
        </li>
        <li className="navElement">
          <Link to="/Wiki">Wiki</Link>
        </li>
        <li className="navElement">
          <Link to="/Feedback">Feedback</Link>
        </li>
        <li className="navElement">
          <a
            href="https://buy.stripe.com/9AQ3e4eO81X57y8aEG"
            target="_blank"
            rel="noopener noreferrer"
          >
            Donate
          </a>
        </li>
      </ul>
    </div>
  );
};
export default Navbar;
