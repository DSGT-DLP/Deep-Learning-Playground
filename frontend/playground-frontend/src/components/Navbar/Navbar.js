import React from "react";
import "./Nav.css";
import { Link } from "react-router-dom";
import DSGTLogo from "../../images/logos/dsgt-logo-light.png";
import { LAYOUT } from "../../constants";

const Navbar = () => {
  return (
    <div id="nav-bar">
      <ul className="nav">
        <li id="title-name">
          <img src={DSGTLogo} alt="DSGT Logo" width="60" height="60" />
          <div style={{ marginRight: 10 }} />
          Deep Learning Playground
        </li>
        <li className="navElement">
          <Link to="/">Home</Link>
        </li>
        <li className="navElement">
          <Link to="/About">About</Link>
        </li>
        <li className="navElement">
          <Link to="/Wiki">Wiki</Link>
        </li>
      </ul>
    </div>
  );
};
export default Navbar;
