import React, { useState, useEffect } from "react";
import DSGTLogo from "../../images/logos/dlp_branding/dlp-logo.png";
import { Link, useNavigate } from "react-router-dom";
import { signOut, onAuthStateChanged } from "firebase/auth";
import { auth } from "../../firebase";
import { toast } from "react-toastify";
import { useDispatch, useSelector } from "react-redux";
import { setCurrentUser } from "../../redux/userLogin";

const AccountButton = () => {
  const navigate = useNavigate();
  const [showDropdown, setShowDropdown] = useState(false);
  const userEmail = useSelector((state) => state.currentUser.email);
  const dispatch = useDispatch();

  
  const goToLogin = () => {
    if (!window.location.href.match(/(\/login$|\/login#$)/g)) {
      // Go to Login page if we aren't already there
      window.location.href = "/login";
    }
  };

  const logout = () => {
    signOut(auth)
      .then(() => {
        dispatch(setCurrentUser(null));
        toast.success("Logged out successfully", { autoClose: 1000 });
      })
      .catch((error) => toast.error(`Error: ${error.code}`));
  };

  useEffect(() => {
    onAuthStateChanged(auth, (user) => {
      if (!user) return;
      const userData = {
        email: user.email,
        uid: user.uid,
        displayName: user.displayName,
        emailVerified: user.emailVerified,
      };
      dispatch(setCurrentUser(userData));
    });
  }, []);

  if (userEmail) {
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
            <button className="accountButton" onClick={logout}>
              Log out
            </button>
          </div>
        )}
      </div>
    );
  } else {
    return (
      <button className="loginButton" onClick={goToLogin}>
        Log in
      </button>
    );
  }
};

const Navbar = () => {
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
          <AccountButton />
        </li>
      </ul>
    </div>
  );
};

export default Navbar;
