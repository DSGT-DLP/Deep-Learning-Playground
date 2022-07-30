import React from "react";
import { Link } from "react-router-dom";
import { PropTypes } from "prop-types";

const SubMenu = (props) => {
  const { title, dropDown } = props;

  return (
    <ul style={{display: dropDown ? "flex" : "none"}}>
        {title.map((title, index) => (
            <li key={index} className="dropdown-content">
                <Link to="/#">
                    {title}
                </Link>
            </li>
        ))}
    </ul>
  )
};

SubMenu.propTypes = {
  title: PropTypes.any,
  dropDown: PropTypes.any,
}

export default SubMenu;