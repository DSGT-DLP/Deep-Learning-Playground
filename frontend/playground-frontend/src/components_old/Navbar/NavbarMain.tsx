import { FormControlLabel, Switch } from "@mui/material";
import { onAuthStateChanged, signOut } from "firebase/auth";
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
import { auth } from "../../firebase";
import DSGTLogo from "../../images/logos/dlp_branding/dlp-logo.png";
import { useAppDispatch, useAppSelector } from "../../redux/hooks";
import { UserType, setCurrentUser } from "../../redux/userLogin";
import { deleteCookie, setCookie } from "../helper_functions/Cookie";
import { LinkContainer } from "react-router-bootstrap";

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

  const goToLogin = () => {
    if (!window.location.href.match(/\/login/g)) {
      // Go to Login page if we aren't already there
      //navigate("/login");
    }
  };

  const logout = () => {
    signOut(auth)
      .then(() => {
        dispatch(setCurrentUser());
        toast.success("Logged out successfully", { autoClose: 1000 });
        deleteCookie("userEmail");
        //navigate("/login");
      })
      .catch((error) => toast.error(`Error: ${error.code}`));
  };

  useEffect(() => {
    onAuthStateChanged(auth, (user) => {
      if (!user?.uid) return;

      const email = user.email || user.providerData[0].email;
      if (!email) throw new Error("No email found");

      const userData: UserType = {
        email: email,
        uid: user.uid,
        displayName: user.displayName ?? "",
        emailVerified: user.emailVerified,
      };
      dispatch(setCurrentUser(userData));
      setCookie("userEmail", user.email);
    });
  }, []);

  if (!user) {
    return <></>;
  }
  return (
    <ThemeProvider theme={theme}>
      <GlobalStyle />
      <Navbar id="navbar-main" className="p-0" expand="lg">
        <Container fluid className="ms-1 pe-0">
          <Navbar.Brand
            href="/"
            className="d-flex align-items-center logo-title"
          >
            <img
              src={DSGTLogo}
              className="logo d-inline-block align-top me-3"
              alt="DSGT Logo"
            />
            Deep Learning Playground
          </Navbar.Brand>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="ms-auto">
              {user?.email ? (
                <LinkContainer to="/train">
                  <Nav.Link>Train</Nav.Link>
                </LinkContainer>
              ) : null}
              <LinkContainer to="/about">
                <Nav.Link>About</Nav.Link>
              </LinkContainer>
              <LinkContainer to="/wiki">
                <Nav.Link>Wiki</Nav.Link>
              </LinkContainer>
              <LinkContainer to="/feedback">
                <Nav.Link>Feedback</Nav.Link>
              </LinkContainer>
              <Nav.Link href={URLs.donate}>Donate</Nav.Link>
              {user?.email ? (
                <NavDropdown title="Account" id="basic-nav-dropdown">
                  <LinkContainer to="/">
                    <NavDropdown.Item href="/">Dashboard</NavDropdown.Item>
                  </LinkContainer>
                  <LinkContainer to="/account-settings">
                    <NavDropdown.Item>Settings</NavDropdown.Item>
                  </LinkContainer>
                  <LinkContainer to="/learn-mod">
                    <NavDropdown.Item>Learn</NavDropdown.Item>
                  </LinkContainer>
                  <NavDropdown.Divider />
                  <LinkContainer to={""}>
                    <NavDropdown.Item onClick={logout}>
                      Log out
                    </NavDropdown.Item>
                  </LinkContainer>
                </NavDropdown>
              ) : (
                <LinkContainer to="">
                  <Nav.Link onClick={goToLogin}>Log in</Nav.Link>
                </LinkContainer>
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
