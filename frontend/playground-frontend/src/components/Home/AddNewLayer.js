import React from "react";
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
      ref={drop}
      className="text-center d-flex justify-content-center align-items-center layer-box add-new-layer-bin"
      style={{ ...styles.container, backgroundColor }}
      data-testid="dustbin"
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
