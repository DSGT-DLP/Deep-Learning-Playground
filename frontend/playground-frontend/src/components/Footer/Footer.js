import React from "react";
import "./Footer.css";
import { Link } from "react-router-dom";
import DSGTLogo from "../../images/logos/dsgt-logo-light.png";

// URLs for now
// https://datasciencegt.org/
// https://www.linkedin.com/company/dsgt/about/
// https://www.instagram.com/datasciencegt/

const Footer = () => {
  return (
    <div className="Footer">
      <div id="footer-name">
        <img src={DSGTLogo} alt="DSGT Logo"/>
        Deep Learning Playground
      </div>
      <div className="footElement">
        <Link to="/">DSGT</Link>
      </div>
      <div className="footElement">
        <Link to="/">Linkedin</Link>
      </div>
      <div className="footElement">
        <Link to="/Wiki">Youtube</Link>
      </div>
      <div className="footElement">
        <Link to="">Instagram</Link>
      </div>
      <div className="footElement">
        Contact us
      </div>
      <div id="navElement">
        <a href="https://github.com/karkir0003/Deep-Learning-Playground">
          <FaGithub size="60" />
        </a>
      </div>
    </div>
  );
};
export default Footer;