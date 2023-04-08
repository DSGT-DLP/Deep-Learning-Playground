import { FaGithub, FaInstagram, FaLinkedin, FaYoutube } from "react-icons/fa";
import { COLORS, URLs } from "../../constants";
import { IconContext } from "react-icons";
import React from "react";

const Footer = () => {
  return (
    <div id="footer" data-testid="footer">
      <IconContext.Provider
        value={{
          color: COLORS.dark_blue,
          size: "2.0rem",
        }}
      >
        <div
          className="footer-element"
          id="footer-socials"
          data-testid="footer-socials"
        >
          <a
            className="foot-element"
            data-testid="linkedin-icon"
            title="Link to LinkedIn profile"
            href={URLs.linkedin}
          >
            <FaLinkedin />
          </a>
          <a
            className="foot-element"
            title="Link to YouTube channel"
            data-testid="youtube-icon"
            href={URLs.youtube}
          >
            <FaYoutube />
          </a>
          <a
            className="foot-element"
            title="Link to Instagram profile"
            data-testid="instagram-icon"
            href={URLs.instagram}
          >
            <FaInstagram />
          </a>
          <a
            className="foot-element"
            title="Link to GitHub repository"
            data-testid="github-icon"
            href={URLs.github}
          >
            <FaGithub />
          </a>
        </div>
      </IconContext.Provider>
      <div
        className="footer-element"
        id="footer-name"
        data-testid="footer-name"
      >
        <p className="copyright">
          Deep Learning Playground © {new Date().getFullYear()}
        </p>
      </div>
    </div>
  );
};

export default Footer;
