import { FormControlLabel, Switch } from "@mui/material";
import storage from "local-storage-fallback";
import React, { useEffect, useState } from "react";
import Container from "react-bootstrap/Container";
import Nav from "react-bootstrap/Nav";
import NavDropdown from "react-bootstrap/NavDropdown";
import Navbar from "react-bootstrap/Navbar";
import { toast } from "react-toastify";
import { ThemeProvider } from "styled-components";
import GlobalStyle from "../../GlobalStyle";
import { URLs } from "../../constants";
import DSGTLogo from "/public/images/logos/dlp_branding/dlp-logo.png";
import { useAppDispatch, useAppSelector } from "@/common/redux/hooks";
import {
  UserType,
  isSignedIn,
  setCurrentUser,
  signOutUser,
} from "@/common/redux/userLogin";
import Image from "next/image";
import Link from "next/link";

const NavbarMain = () => {
  const user = useAppSelector((state) => state.currentUser.user);
  const dispatch = useAppDispatch();
  function getInitialTheme() {
    const savedTheme = storage.getItem("theme");
    return savedTheme
      ? JSON.parse(savedTheme)
      : { mode: "light", checked: false };
  }

  const [theme, setTheme] = useState(getInitialTheme);

  useEffect(() => {
    storage.setItem("theme", JSON.stringify(theme));
  }, [theme]);

  const toggleTheme = () => {
    if (theme.mode === "light") {
      setTheme({ mode: "dark", checked: true });
    } else {
      setTheme({ mode: "light", checked: false });
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <GlobalStyle />
      <Navbar id="navbar-main" className="p-0" expand="lg">
        <Container fluid className="ms-1 pe-0">
          <Link href="/" passHref legacyBehavior>
            <Navbar.Brand
              className="d-flex align-items-center logo-title"
              style={{
                position: "relative",
              }}
            >
              <div
                style={{
                  position: "relative",
                  height: "50px",
                  width: "50px",
                  paddingLeft: "60px",
                }}
              >
                <Image
                  src={DSGTLogo}
                  className="logo d-inline-block align-top me-3"
                  alt="DSGT Logo"
                  fill={true}
                  style={{ objectFit: "contain", margin: "auto" }}
                />
              </div>
              Deep Learning Playground
            </Navbar.Brand>
          </Link>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="ms-auto">
              {isSignedIn(user) ? (
                <Link href="/train" passHref legacyBehavior>
                  <Nav.Link>Train</Nav.Link>
                </Link>
              ) : null}
              <Link href="/about" passHref legacyBehavior>
                <Nav.Link>About</Nav.Link>
              </Link>
              <Link href="/wiki" passHref legacyBehavior>
                <Nav.Link>Wiki</Nav.Link>
              </Link>
              <Link href="/feedback" passHref legacyBehavior>
                <Nav.Link>Feedback</Nav.Link>
              </Link>
              <Nav.Link href={URLs.donate}>Donate</Nav.Link>
              {isSignedIn(user) ? (
                <NavDropdown title="Account" id="basic-nav-dropdown">
                  <Link href="/" passHref legacyBehavior>
                    <NavDropdown.Item>Dashboard</NavDropdown.Item>
                  </Link>
                  <Link href="/account-settings" passHref legacyBehavior>
                    <NavDropdown.Item>Settings</NavDropdown.Item>
                  </Link>
                  <Link href="/learn-mod" passHref legacyBehavior>
                    <NavDropdown.Item>Learn</NavDropdown.Item>
                  </Link>
                  <NavDropdown.Divider />
                  <Link href={""} passHref legacyBehavior>
                    <NavDropdown.Item
                      onClick={() => {
                        dispatch(signOutUser());
                      }}
                    >
                      Log out
                    </NavDropdown.Item>
                  </Link>
                </NavDropdown>
              ) : (
                <Link href="/login" passHref legacyBehavior>
                  <Nav.Link>Log in</Nav.Link>
                </Link>
              )}
            </Nav>
          </Navbar.Collapse>
        </Container>
        <FormControlLabel
          control={
            <Switch
              id="mode-switch"
              onChange={toggleTheme}
              checked={theme.checked}
            ></Switch>
          }
          label={`${theme.mode === "dark" ? "🌙" : "☀️"}`}
        />
      </Navbar>
    </ThemeProvider>
  );
};

export default NavbarMain;
