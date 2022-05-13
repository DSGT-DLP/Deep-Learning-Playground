import React from "react";
import PropTypes from "prop-types";
import RectContainer from "./RectContainer";
import { COLORS, GENERAL_STYLES, ITEM_TYPES } from "../constants";
import { useDrag } from "react-dnd";

const LayerChoice = (props) => {
  //   const [{ isDragging }, drag] = useDrag(() => ({
  //     type: ITEM_TYPES.NEW_LAYER,
  //     item: props.text,
  //     end: (item, monitor) => {
  //       const dropResult = monitor.getDropResult();
  //       if (item && dropResult) {
  //         alert(`You dropped ${item.name} into ${dropResult.name}!`);
  //       }
  //     },
  //     collect: (monitor) => ({
  //       isDragging: monitor.isDragging(),
  //       handlerId: monitor.getHandlerId(),
  //     }),
  //   }));
  //   const opacity = isDragging ? 0.4 : 1;
  return (
    <RectContainer
    //   ref={drag}
      style={{ backgroundColor: COLORS.addLayer }}
    //   data-testid={`box`}
    >
      <p style={{ ...GENERAL_STYLES.p, color: COLORS.layer, fontSize: 25 }}>
        {props.text}
      </p>
    </RectContainer>
  );
};

LayerChoice.propTypes = {
  text: PropTypes.string.isRequired,
};

export default LayerChoice;
