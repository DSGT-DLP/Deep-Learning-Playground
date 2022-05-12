import React from "react";
import PropTypes from "prop-types";
import RectContainer from "./RectContainer";
import { COLORS, ITEM_TYPES } from "../constants";
import { useDrop } from "react-dnd";

const AddNewLayer = (props) => {
  return (
    <RectContainer style={styles.container}>
      <p style={{ fontSize: 50, fontWeight: "1000", color: COLORS.layer }}>+</p>
    </RectContainer>
  );
};

AddNewLayer.propTypes = {};

export default AddNewLayer;

const styles = {
  container: {
    border: `5px dashed ${COLORS.layer}`,
    backgroundColor: COLORS.addLayer,
  },
};
