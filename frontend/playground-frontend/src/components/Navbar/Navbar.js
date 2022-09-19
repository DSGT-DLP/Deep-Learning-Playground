import DSGTLogo from "../../images/logos/dlp_branding/dlp-logo.png";
import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import PropTypes from "prop-types";
import { auth } from "../../firebase";
import { useAuthState } from "react-firebase-hooks/auth";

const AccountButton = ({ setShowLogin }) => {
  const navigate = useNavigate();
  const [user] = useAuthState(auth);
  const [showDropdown, setShowDropdown] = useState(false);

  if (user) {
    return (
      <div id="accountButtonWrapper">
        <button
          className="loginButton accountButtonMain"
          onClick={() => setShowDropdown((prev) => !prev)}
        >
          Account
        </button>
        {showDropdown && (
          <div id="accountButtons">
            <button className="accountButton" onClick={() => navigate("/")}>
              Dashboard
            </button>
            <button className="accountButton">Settings</button>
            <button className="accountButton">Learn</button>
            <button className="accountButton" onClick={() => auth.signOut()}>
              Log out
            </button>
          </div>
        )}
      </div>
    );
  } else {
    return (
      <button className="loginButton" onClick={() => setShowLogin(true)}>
        Log in
      </button>
    );
  }
};

const Navbar = ({ setShowLogin }) => {
  return (
    <div className="header-footer" id="nav-bar">
      <Link to="/" className="image-title">
        <img src={DSGTLogo} alt="DSGT Logo" width="60" height="60" />
        <div style={{ marginRight: 10 }} />
        Deep Learning Playground
      </Link>
      <ul className="nav">
        <li id="title-name"></li>

        <li className="navElement">
          <Link to="/train">Train</Link>
        </li>

        <li className="navElement">
          <Link to="/about">About</Link>
        </li>
        <li className="navElement">
          <Link to="/wiki">Wiki</Link>
        </li>
        <li className="navElement">
          <Link to="/feedback">Feedback</Link>
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
        <li className="navElement">
          <AccountButton setShowLogin={setShowLogin} />
        </li>
      </ul>
    </div>
  );
};
export default Navbar;

AccountButton.propTypes = {
  setShowLogin: PropTypes.func.isRequired,
};

Navbar.propTypes = {
  setShowLogin: PropTypes.func.isRequired,
};
