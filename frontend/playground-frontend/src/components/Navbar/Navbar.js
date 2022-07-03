import React from "react";
import "./Nav.css";
import { Link } from "react-router-dom";
import DSGTLogo from "../../images/logos/dsgt-logo-light.png";
import { FaGithub } from "react-icons/fa";
import { FaYoutube } from "react-icons/fa";
import { LAYOUT } from "../../constants";

const Navbar = () => {
    return (
        <div id="nav-bar">
            <div className="image-title" style={{ fontSize: "25px" }}>
                <img src={DSGTLogo} alt="DSGT Logo" width="61" height="60" />
                <div style={{ marginRight: 10 }} />
                Deep Learning Playground
            </div>
            <ul className="nav">
                <li id="navElement">
                    <a href="https://github.com/karkir0003/Deep-Learning-Playground">
                        <FaGithub size="60" />
                    </a>
                </li>
                <li id="title-name"></li>
                <li className="navElement">
                    <Link to="/">Home</Link>
                </li>
                <li className="navElement">
                    <Link to="/About">About</Link>
                </li>
                <li className="navElement">
                    <Link to="/Wiki">Wiki</Link>
                </li>
                <li className="navElement">
                    <Link to="/Feedback">Feedback </Link>
                </li>
            </ul>
        </div>
    );
};
export default Navbar;
