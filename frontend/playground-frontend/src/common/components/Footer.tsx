import React from "react";
import { COLORS, URLs } from "../../constants";
import { LinkedIn, YouTube, Instagram, GitHub } from "@mui/icons-material";
import { ThemeProvider, createTheme } from "@mui/material/styles";

const theme = createTheme({
  components: {
    MuiSvgIcon: {
      styleOverrides: {
        root: {
          fontSize: 40,
          color: COLORS.dark_blue
        },
      },
    },
  },
});

const Footer = () => {
  return (
    <div id="footer" data-testid="footer" className="flex-wrapper">
      <ThemeProvider theme={theme}>
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
            <LinkedIn />
          </a>
          <a
            className="foot-element"
            title="Link to YouTube channel"
            data-testid="youtube-icon"
            href={URLs.youtube}
          >
            <YouTube />
          </a>
          <a
            className="foot-element"
            title="Link to Instagram profile"
            data-testid="instagram-icon"
            href={URLs.instagram}
          >
            <Instagram />
          </a>
          <a
            className="foot-element"
            title="Link to GitHub repository"
            data-testid="github-icon"
            href={URLs.github}
          >
            <GitHub />
          </a>
        </div>
      </ThemeProvider>
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
