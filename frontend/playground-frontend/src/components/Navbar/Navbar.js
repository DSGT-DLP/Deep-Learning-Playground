import React from "react";
import "./navbar.css";
import { Link } from "react-router-dom";
const Navbar = () => {
  return (
    <ul>
      <li id="Home">
        <Link to="/">Home</Link>
      </li>
      <li>
        <Link to="/About">About</Link>
      </li>
    </ul>
  );
};
export default Navbar;
{
  /*
const styles = {
  navList: {
    fontSize: 25,
    paddingRight: 25,
    display: "inline",
    alignItems: "left",
    textDecoration: "none",
    fontFamily: "verdona",
    fontWeigh: "bold",
  },
};
*/
}
