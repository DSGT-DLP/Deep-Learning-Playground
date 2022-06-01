import React from "react";
import "./Nav.css";
import { Link } from "react-router-dom";
const Navbar = () => {
  return (
    <div>
      <p>Deep Learning</p>
      <ul className="nav">
        <li className="navList">
          <Link to="/">Home</Link>
        </li>
        <li className="navList">
          <Link to="/About">About</Link>
        </li>
      </ul>
    </div>
  );
};
export default Navbar;
