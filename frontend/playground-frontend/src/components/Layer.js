import React from "react";
import PropTypes from "prop-types";
import RectContainer from "./RectContainer";
import { COLORS, GENERAL_STYLES } from "../constants";

const Layer = (props) => {
  return (
    <RectContainer style={{ backgroundColor: COLORS.layer }}>
      <p style={{ ...GENERAL_STYLES.p, color: COLORS.white, fontSize: 30 }}>
        Linear
      </p>
    </RectContainer>
  );
};

Layer.propTypes = {};

export default Layer;
