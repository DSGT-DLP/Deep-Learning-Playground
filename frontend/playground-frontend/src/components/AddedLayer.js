import React from "react";
import PropTypes from "prop-types";
import RectContainer from "./RectContainer";
import { COLORS, GENERAL_STYLES } from "../constants";
import { useDrop } from "react-dnd";

const AddedLayer = (props) => {
  return (
    <RectContainer style={{ backgroundColor: COLORS.layer }}>
      <p style={{ ...GENERAL_STYLES.p, color: COLORS.white, fontSize: 25 }}>
        Linear
      </p>
    </RectContainer>
  );
};

AddedLayer.propTypes = {};

export default AddedLayer;
