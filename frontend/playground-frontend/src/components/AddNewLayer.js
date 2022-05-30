import React from "react";
import PropTypes from "prop-types";
import RectContainer from "./RectContainer";
import { COLORS, ITEM_TYPES } from "../constants";
import { useDrop } from "react-dnd";

const AddNewLayer = (props) => {
  const [{ canDrop, isOver }, drop] = useDrop(() => ({
    accept: ITEM_TYPES.NEW_LAYER,
    drop: () => ({ name: "AddNewLayer" }),
    collect: (monitor) => ({
      isOver: monitor.isOver(),
      canDrop: monitor.canDrop(),
    }),
  }));
  const isActive = canDrop && isOver;
  let backgroundColor = COLORS.addLayer;
  if (isActive) {
    backgroundColor = COLORS.layer;
  } else if (canDrop) {
    backgroundColor = "white";
  }
  return (
    <RectContainer
      ref2={drop}
      style={{ ...styles.container, backgroundColor }}
      dataTestid="dustbin"
    >
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
