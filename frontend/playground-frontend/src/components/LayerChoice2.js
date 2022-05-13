import { useDrag } from "react-dnd";
import { ITEM_TYPES } from "../constants";

const LayerChoice2 = ({ name }) => {
  const [{ isDragging }, drag] = useDrag(() => ({
    type: ITEM_TYPES.NEW_LAYER,
    item: { name },
    end: (item, monitor) => {
      const dropResult = monitor.getDropResult();
      if (item && dropResult) {
        alert(`You dropped ${item.name} into ${dropResult.name}!`);
      }
    },
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
      handlerId: monitor.getHandlerId(),
    }),
  }));
  const opacity = isDragging ? 0.4 : 1;
  return (
    <div ref={drag} style={{ ...style, opacity }} data-testid={`box`}>
      {name}
    </div>
  );
};

export default LayerChoice2;

const style = {
  border: "5px dashed #CD7BFF",
  backgroundColor: "white",
  padding: "0.5rem 1rem",
  marginRight: "1.5rem",
  marginBottom: "1.5rem",
  cursor: "move",
  float: "left",
};