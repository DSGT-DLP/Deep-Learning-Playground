import React from "react";
import PropTypes from "prop-types";
import RectContainer from "./RectContainer";
import { COLORS, GENERAL_STYLES } from "../constants";

const AddedLayer = (props) => {
  return (
    <RectContainer style={{ backgroundColor: COLORS.layer }}>
      <p style={{ ...GENERAL_STYLES.p, color: COLORS.white, fontSize: 25 }}>
        {props.text}
      </p>
    </RectContainer>
  );
};

AddedLayer.propTypes = {
  text: PropTypes.string.isRequired,
};

export default AddedLayer;
