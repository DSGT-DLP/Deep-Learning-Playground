import React from "react";
import "./Nav.css";
import { Link } from "react-router-dom";
const Navbar = () => {
  return (
    <ul className="nav">
      <li className="navList">
        <Link to="/">Home</Link>
      </li>
      <li className="navList">
        <Link to="/About">About</Link>
      </li>
    </ul>
  );
};
export default Navbar;
