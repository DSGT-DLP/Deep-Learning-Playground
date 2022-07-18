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
          <Link to="/Feedback">Feedback </Link>
        </li>
        <li className="navElement" style={{ height: 32, paddingTop: 17 }}>
          <form
            action="https://www.paypal.com/donate"
            method="post"
            target="_top"
          >
            <input
              type="hidden"
              name="hosted_button_id"
              value="Y7VZR7PVRLTPE"
            />
            <input
              type="image"
              src="https://www.paypalobjects.com/en_US/i/btn/btn_donate_LG.gif"
              border="0"
              name="submit"
              title="PayPal - The safer, easier way to pay online!"
              style={{ border: "none" }}
              alt="Donate with PayPal button"
            />
            <img
              alt=""
              border="0"
              src="https://www.paypal.com/en_US/i/scr/pixel.gif"
              width="1"
              height="1"
            />
          </form>
        </li>
      </ul>
    </div>
  );
};
export default Navbar;
