import React from "react";
import RectContainer from "./RectContainer";
import { COLORS, ITEM_TYPES } from "../../constants";
import { useDrop } from "react-dnd";

const AddNewLayer = () => {
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
    <div
      ref2={drop}
      className="text-center d-flex justify-content-center align-items-center layer-box add-new-layer"
      style={{ ...styles.container, backgroundColor }}
      dataTestid="dustbin"
    >
      +
    </div>
  );
};

export default AddNewLayer;

const styles = {
  container: {
    border: `5px dashed ${COLORS.layer}`,
    backgroundColor: COLORS.addLayer,
  },
};
