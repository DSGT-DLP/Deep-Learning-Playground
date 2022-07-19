import React from "react";
import "../../App.css";
import {
  FaGithub,
  FaInstagram,
  FaLinkedin,
  FaYoutube,
} from "react-icons/fa";
import { URLs } from "../../constants";
import { IconContext } from "react-icons";

const Footer = () => {
  return (
    <>
    <IconContext.Provider value={{color: "white", size: "2.5rem", className:"foot-element"}}>
      <div className="header-footer" id="footer">
        <div className="foot-element">
          <a href={URLs.linkedin}>
            <FaLinkedin />
          </a>
        </div>
        <div className="foot-element">
          <a href={URLs.youtube}>
            <FaYoutube />
          </a>
        </div>
        <div className="foot-element">
          <a href={URLs.instagram}>
            <FaInstagram />
          </a>
        </div>
        <div className="foot-element">
          <a href={URLs.github}>
            <FaGithub />
          </a>
        </div>
      </div>
      </IconContext.Provider>
      <div className="header-footer" id="footer-name">
        Deep Learning Playground © {new Date().getFullYear()}
      </div>
    </>
  );
};

export default Footer;