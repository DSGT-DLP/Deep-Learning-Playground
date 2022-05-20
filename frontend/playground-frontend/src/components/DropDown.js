import React, { useState } from "react";
import PropTypes from "prop-types";
import Select from "react-select";
import { GENERAL_STYLES } from "../constants";

const DropDown = (props) => {
  return (
    <Select
      options={props.options}
      onChange={props.onChange}
      styles={dropdownStyes}
      defaultValue={props.defaultValue}
    />
  );
};

DropDown.propTypes = {
  options: PropTypes.arrayOf(PropTypes.object),
  onChange: PropTypes.func,
  defaultValue: PropTypes.oneOfType([
    PropTypes.object,
    PropTypes.number,
    PropTypes.string,
  ]),
};

export default DropDown;

const dropdownStyes = {
  control: (base) => ({
    ...base,
    ...GENERAL_STYLES.p,
    border: "none",
    backgroundColor: "transparent",
    fontSize: 18,
  }),
  menu: (base) => ({
    ...base,
    ...GENERAL_STYLES.p,
    fontSize: 18,
  }),
};
