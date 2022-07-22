import React from "react";
import "../../App.css";
import { FaGithub, FaInstagram, FaLinkedin, FaYoutube } from "react-icons/fa";
import { COLORS, URLs } from "../../constants";
import { IconContext } from "react-icons";

const Footer = () => {
  return (
    <>
      <IconContext.Provider
        value={{
          color: COLORS.dark_blue,
          size: "2.0rem",
        }}
      >
        <div className="header-footer" id="footer">
          <a className="foot-element" href={URLs.linkedin}>
            <FaLinkedin />
          </a>
          <a className="foot-element" href={URLs.youtube}>
            <FaYoutube />
          </a>
          <a className="foot-element" href={URLs.instagram}>
            <FaInstagram />
          </a>
          <a className="foot-element" href={URLs.github}>
            <FaGithub />
          </a>
        </div>
      </IconContext.Provider>
      <div className="header-footer" id="footer-name">
        Deep Learning Playground © {new Date().getFullYear()}
      </div>
    </>
  );
};

export default Footer;
