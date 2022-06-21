import React from "react";
import "./Footer.css";
import { Link } from "react-router-dom";
import DSGTLogo from "../../images/logos/dsgt-logo-light.png";
import { FaGithub, FaCopyright } from "react-icons/fa";
import {BsInstagram, BsLinkedin, BsYoutube} from "react-icons/bs";
// import {AiFillYoutube, AiFillLinkedin} from "react-icons/AiFill";

// URLs for now
// https://datasciencegt.org/
// https://www.linkedin.com/company/dsgt/about/
// https://www.instagram.com/datasciencegt/

const Footer = () => {
  return (
    <div className="Footer">
      <div id="footer-name">
        <img src={DSGTLogo} alt="DSGT Logo" width="60" height="60"/>
        Deep Learning Playground
      </div>
      {/* <div className="footElement">
        <a href="https://datasciencegt.org/">
          <AiFillLinkedin size="60" />
        </a>
      </div> */}
      <div className="footElement">
        <a href="https://www.linkedin.com/company/dsgt/about/">
          <BsLinkedin size="60" />
        </a>
      </div>
      <div className="footElement">
        <a href="www.youtube.com">
          <BsYoutube size="60" />
        </a>
      </div>
      <div className="footElement">
        <a href="https://www.instagram.com/datasciencegt/">
          <BsInstagram size="60" />
        </a>
      </div>
      <div className="footElement">
        Contact us
      </div>
      <div id="navElement">
        <a href="https://github.com/karkir0003/Deep-Learning-Playground">
          <FaGithub size="60" />
        </a>
      </div>
      <div id="navElement">
      <FaCopyright size="60"/>
      </div>
    </div>
  );
};
export default Footer;