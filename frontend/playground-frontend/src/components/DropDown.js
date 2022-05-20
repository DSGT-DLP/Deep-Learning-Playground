import React from "react";
import PropTypes from "prop-types";
import Select from "react-dropdown-select";
import { GENERAL_STYLES } from "../constants";

const DropDown = (props) => {
  return (
    <Select
      options={props.options}
      onChange={props.onChange}
      style={{ ...GENERAL_STYLES.p, border: "none" }}
      sortBy="label"
    />
  );
};

DropDown.propTypes = {
  options: PropTypes.arrayOf(PropTypes.object).isRequired,
  onChange: PropTypes.func,
};

export default DropDown;
