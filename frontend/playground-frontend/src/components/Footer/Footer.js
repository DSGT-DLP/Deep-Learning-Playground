import React from "react";
import { FaGithub, FaInstagram, FaLinkedin, FaYoutube } from "react-icons/fa";
import { COLORS, URLs } from "../../constants";
import { IconContext } from "react-icons";

const Footer = () => {
  return (
    <div id="footer">
      <IconContext.Provider
        value={{
          color: COLORS.dark_blue,
          size: "2.0rem",
        }}
      >
        <div className="header-footer" id="footer-socials">
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
        <p className="copyright">
          Deep Learning Playground © {new Date().getFullYear()}
        </p>
      </div>
    </div>
  );
};

export default Footer;
