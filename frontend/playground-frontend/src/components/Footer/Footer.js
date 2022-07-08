import React from "react";
import "../../App.css";
import {
  FaGithub,
  FaInstagram,
  FaLinkedin,
  FaYoutube,
} from "react-icons/fa";
import { URLs } from "../../constants";

const Footer = () => {
  return (
    <>
      <div className="header-footer" id="footer">
        <div className="foot-element">
          <a href={URLs.linkedin}>
            <FaLinkedin size="60" />
          </a>
        </div>
        <div className="foot-element">
          <a href={URLs.youtube}>
            <FaYoutube size="60" />
          </a>
        </div>
        <div className="foot-element">
          <a href={URLs.instagram}>
            <FaInstagram size="60" />
          </a>
        </div>
        <div className="foot-element">
          <a href={URLs.github}>
            <FaGithub size="60" />
          </a>
        </div>
      </div>
      <div className="header-footer" id="footer-name">
        Deep Learning Playground © {new Date().getFullYear()}
      </div>
    </>
  );
};

export default Footer;