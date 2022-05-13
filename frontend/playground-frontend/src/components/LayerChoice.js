import React from "react";
import PropTypes from "prop-types";
import RectContainer from "./RectContainer";
import { COLORS, GENERAL_STYLES, ITEM_TYPES } from "../constants";
import { useDrag } from "react-dnd";

const LayerChoice = (props) => {
  const [{ isDragging }, drag] = useDrag(() => ({
    type: ITEM_TYPES.NEW_LAYER,
    item: props.text,
    end: (item, monitor) => {
      const dropResult = monitor.getDropResult();
      if (item && dropResult) {
        props.onDrop(props.text);
        // alert(`You dropped ${props.text} into ${dropResult.name}!`);
      }
    },
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
      handlerId: monitor.getHandlerId(),
    }),
  }));
  const opacity = isDragging ? 0.4 : 1;
  return (
    <RectContainer
      ref2={drag}
      style={{ backgroundColor: COLORS.addLayer }}
      dataTestid={`box`}
    >
      <p style={{ ...styles.text, opacity }}>{props.text}</p>
    </RectContainer>
  );
};

LayerChoice.propTypes = {
  text: PropTypes.string.isRequired,
  onDrop: PropTypes.func,
};

export default LayerChoice;

const styles = {
  text: {
    ...GENERAL_STYLES.p,
    color: COLORS.layer,
    fontSize: 25,
  },
};
