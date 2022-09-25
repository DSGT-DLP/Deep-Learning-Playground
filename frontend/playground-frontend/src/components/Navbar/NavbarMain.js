import React, { useState, useEffect } from "react";
import DSGTLogo from "../../images/logos/dlp_branding/dlp-logo.png";
import { Link, useNavigate } from "react-router-dom";
import { signOut, onAuthStateChanged } from "firebase/auth";
import { auth } from "../../firebase";
import { toast } from "react-toastify";
import { useDispatch, useSelector } from "react-redux";
import { setCurrentUser } from "../../redux/userLogin";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Nav from "react-bootstrap/Nav";
import Navbar from "react-bootstrap/Navbar";
import NavDropdown from "react-bootstrap/NavDropdown";
import { URLs } from "../../constants";

// const AccountButton = () => {
//   const navigate = useNavigate();

//   if (userEmail) {
//     return (
//       <div id="accountButtonWrapper">
//         <button
//           className="loginButton accountButtonMain"
//           onClick={() => setShowDropdown((prev) => !prev)}
//         >
//           Account
//         </button>
//         {showDropdown && (
//           <div id="accountButtons">
//             <button className="accountButton" onClick={() => navigate("/")}>
//               Dashboard
//             </button>
//             <button className="accountButton">Settings</button>
//             <button className="accountButton">Learn</button>
//             <button className="accountButton" onClick={logout}>
//               Log out
//             </button>
//           </div>
//         )}
//       </div>
//     );
//   } else {
//     return (
//       <button className="loginButton" onClick={goToLogin}>
//         Log in
//       </button>
//     );
//   }
// };

// const Navbar2 = () => {
//   return (
//     <div className="header-footer" id="nav-bar">
//       <Link to="/" className="image-title">
//         <img src={DSGTLogo} alt="DSGT Logo" width="60" height="60" />
//         <div style={{ marginRight: 10 }} />
//         Deep Learning Playground
//       </Link>
//       <ul className="nav">
//         <li id="title-name"></li>

//         <li className="navElement">
//           <Link to="/train">Train</Link>
//         </li>

//         <li className="navElement">
//           <Link to="/about">About</Link>
//         </li>
//         <li className="navElement">
//           <Link to="/wiki">Wiki</Link>
//         </li>
//         <li className="navElement">
//           <Link to="/feedback">Feedback</Link>
//         </li>
//         <li className="navElement">
//           <a
//             href="https://buy.stripe.com/9AQ3e4eO81X57y8aEG"
//             target="_blank"
//             rel="noopener noreferrer"
//           >
//             Donate
//           </a>
//         </li>
//         <li className="navElement">
//           <AccountButton />
//         </li>
//       </ul>
//     </div>
//   );
// };

const NavbarMain = () => {
  const [showDropdown, setShowDropdown] = useState(false);
  const userEmail = useSelector((state) => state.currentUser.email);
  const dispatch = useDispatch();

  const goToLogin = () => {
    if (!window.location.href.match(/(\/login$|\/login#$)/g)) {
      // Go to Login page if we aren't already there
      window.location.href = "/login";
    }
  };

  const logout = () => {
    signOut(auth)
      .then(() => {
        dispatch(setCurrentUser(null));
        toast.success("Logged out successfully", { autoClose: 1000 });
      })
      .catch((error) => toast.error(`Error: ${error.code}`));
  };

  useEffect(() => {
    onAuthStateChanged(auth, (user) => {
      if (!user) return;
      const userData = {
        email: user.email,
        uid: user.uid,
        displayName: user.displayName,
        emailVerified: user.emailVerified,
      };
      dispatch(setCurrentUser(userData));
    });
  }, []);

  return (
    <Navbar id="navbar-main" className="p-0" expand="lg">
      <Container fluid className="ms-1 pe-0">
        <Navbar.Brand href="/" className="d-flex align-items-center logo-title">
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
            <Nav.Link href="/train">Train</Nav.Link>
            <Nav.Link href="/about">About</Nav.Link>
            <Nav.Link href="/wiki">Wiki</Nav.Link>
            <Nav.Link href="/feedback">Feedback</Nav.Link>
            <Nav.Link href={URLs.donate}>Donate</Nav.Link>
            <Nav.Link href="#" onClick={goToLogin}>
              Log in
            </Nav.Link>
            <NavDropdown title="Account" id="basic-nav-dropdown">
              <NavDropdown.Item href="/">Dashboard</NavDropdown.Item>
              <NavDropdown.Item href="#">Settings</NavDropdown.Item>
              <NavDropdown.Item href="#">Learn</NavDropdown.Item>
              <NavDropdown.Divider />
              <NavDropdown.Item href="#" onClick={logout}>
                Log out
              </NavDropdown.Item>
            </NavDropdown>
          </Nav>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
};

export default NavbarMain;
