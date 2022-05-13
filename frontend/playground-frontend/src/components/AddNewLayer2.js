import { useDrop } from "react-dnd";
import { ITEM_TYPES } from "../constants";

const style = {
  height: "12rem",
  width: "12rem",
  marginRight: "1.5rem",
  marginBottom: "1.5rem",
  color: "white",
  padding: "1rem",
  textAlign: "center",
  fontSize: "1rem",
  lineHeight: "normal",
  float: "left",
};
const AddNewLayer2 = () => {
  const [{ canDrop, isOver }, drop] = useDrop(() => ({
    accept: ITEM_TYPES.NEW_LAYER,
    drop: () => ({ name: "Dustbin" }),
    collect: (monitor) => ({
      isOver: monitor.isOver(),
      canDrop: monitor.canDrop(),
    }),
  }));
  const isActive = canDrop && isOver;
  let backgroundColor = "#222";
  if (isActive) {
    backgroundColor = "darkgreen";
  } else if (canDrop) {
    backgroundColor = "darkkhaki";
  }
  return (
    <div ref={drop} style={{ ...style, backgroundColor }} data-testid="dustbin">
      {isActive ? "Release to drop" : "Drag a box here"}
    </div>
  );
};

export default AddNewLayer2;