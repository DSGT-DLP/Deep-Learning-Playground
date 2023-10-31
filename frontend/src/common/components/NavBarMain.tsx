import { useAppDispatch, useAppSelector } from "@/common/redux/hooks";
import { isSignedIn, signOutUser } from "@/common/redux/userLogin";
import ArrowDropDownIcon from "@mui/icons-material/ArrowDropDown";
import { FormControlLabel, Icon, Switch } from "@mui/material";
import AppBar from "@mui/material/AppBar";
import Button from "@mui/material/Button";
import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";
import MenuList from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Stack from "@mui/material/Stack";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import storage from "local-storage-fallback";
import Image from "next/image";
import Link from "next/link";
import React, { useEffect, useState } from "react";
import { ThemeProvider } from "styled-components";
import { URLs } from "../../constants";
import DSGTLogo from "/public/images/logos/dlp_branding/dlp-logo.png";

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

  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const open = Boolean(anchorEl);

  const handleCloseUserMenu = () => {
    setAnchorEl(null);
  };

  const handleOpenUserMenu = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  return (
    <ThemeProvider theme={theme}>
      <AppBar id="navbar-main" className="p-0" position="static">
        <Container maxWidth="lg">
          <Toolbar
            disableGutters
            sx={{
              display: { xs: "flex" },
              flexDirection: "row",
              justifyContent: "space-between",
            }}
          >
            <Link href="/" className="logo-redirect-link">
              <Stack direction="row" alignItems="center">
                <Icon
                  sx={{ display: { xs: "none", md: "flex" }, ml: -5, mr: 1 }}
                >
                  <Image
                    src={DSGTLogo}
                    alt="DSGT Logo"
                    width={100}
                    height={100}
                  />
                </Icon>
                <Typography
                  noWrap
                  className="d-flex align-items-center logo-title"
                  sx={{
                    mr: 2,
                    display: { xs: "none", md: "flex" },
                    textDecoration: "none",
                    fontWeight: "500",
                    fontSize: "17px",
                  }}
                >
                  Deep Learning Playground
                </Typography>
              </Stack>
            </Link>

            <Grid
              sx={{
                flexGrow: 1,
                display: { xs: "none", md: "flex", justifyContent: "right" },
              }}
            >
              <Grid item>
                {isSignedIn(user) ? (
                  <Link href="/train" passHref className="nav-link">
                    Hello World
                  </Link>
                ) : null}
              </Grid>
              <Grid item>
                <Link href="/about" passHref className="nav-link">
                  About
                </Link>
              </Grid>
              <Grid item>
                <Link href="/wiki" passHref className="nav-link">
                  Wiki
                </Link>
              </Grid>
              <Grid item>
                <Link href="/feedback" passHref className="nav-link">
                  Feedback
                </Link>
              </Grid>
              <Grid item>
                <Link href={URLs.donate} passHref className="nav-link">
                  Donate
                </Link>
              </Grid>
              {isSignedIn(user) ? (
                <Grid item>
                  <div>
                    <Button
                      sx={{
                        my: -0.75,
                        mx: -1,
                      }}
                    >
                      <Typography
                        onClick={handleOpenUserMenu}
                        className="nav-link"
                        style={{
                          fontFamily: "Lato, Arial, Helvetica, sans-serif",
                          textTransform: "none",
                        }}
                      >
                        Account <ArrowDropDownIcon />
                      </Typography>
                    </Button>
                    <MenuList
                      anchorEl={anchorEl}
                      open={open}
                      onClose={handleCloseUserMenu}
                    >
                      <MenuItem>
                        <Link href="/dashboard" id="basic-nav-dropdown">
                          Dashboard
                        </Link>
                      </MenuItem>
                      <MenuItem>
                        <Link href="/settings" id="basic-nav-dropdown">
                          Settings
                        </Link>
                      </MenuItem>
                      <MenuItem divider>
                        <Link href="/learn" id="basic-nav-dropdown">
                          Learn
                        </Link>
                      </MenuItem>
                      <Link href={""} passHref id="basic-nav-dropdown">
                        <MenuItem
                          onClick={() => {
                            dispatch(signOutUser());
                          }}
                        >
                          <Typography
                            sx={{
                              fontFamily: "Lato, Arial, Helvetica, sans-serif",
                            }}
                          >
                            Log out
                          </Typography>
                        </MenuItem>
                      </Link>
                    </MenuList>
                  </div>
                </Grid>
              ) : (
                <Link href="/login" passHref className="nav-link">
                  Log in
                </Link>
              )}
            </Grid>
            <FormControlLabel
              style={{
                marginLeft: "auto",
                marginRight: 0,
                flexDirection: "row-reverse",
              }}
              control={
                <Switch
                  id="mode-switch"
                  onChange={toggleTheme}
                  checked={theme.checked}
                ></Switch>
              }
              label={`${theme.mode === "dark" ? "🌙" : "☀️"}`}
            />
          </Toolbar>
        </Container>
      </AppBar>
    </ThemeProvider>
  );
};

export default NavbarMain;
