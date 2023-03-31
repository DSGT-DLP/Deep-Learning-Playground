import React from "react";
import Select, { GroupBase, Props, StylesConfig } from "react-select";
import { GENERAL_STYLES } from "../../constants";

function DropDown<
  Option,
  IsMulti extends boolean = false,
  Group extends GroupBase<Option> = GroupBase<Option>
>(props: Props<Option, IsMulti, Group>) {
  const { options, onChange, defaultValue, isMulti } = props;

  const dropdownStyles: StylesConfig<Option, IsMulti, GroupBase<Option>> = {
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

  return (
    <Select
      options={options}
      onChange={onChange}
      styles={dropdownStyles}
      defaultValue={defaultValue}
      isMulti={isMulti}
    />
  );
}

export default DropDown;
