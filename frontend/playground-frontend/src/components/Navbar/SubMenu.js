import React, { useState } from "react";
import { Link } from "react-router-dom";

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

export default SubMenu;