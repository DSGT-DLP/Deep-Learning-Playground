import PropTypes from "prop-types";
import React, { useState } from "react";
import RectContainer from "./RectContainer";
import Tooltip, { TooltipProps, tooltipClasses } from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import { COLORS, GENERAL_STYLES, ITEM_TYPES } from "../../constants";
import { styled } from "@mui/material/styles";
import { useDrag } from "react-dnd";
import InfoIcon from "@mui/icons-material/Info";

const LayerChoice = (props) => {
  const { layer, onDrop } = props;

  const [{ isDragging }, drag] = useDrag(() => ({
    type: ITEM_TYPES.NEW_LAYER,
    item: props.layer,
    end: (item, monitor) => {
      const dropResult = monitor.getDropResult();
      if (item && dropResult) {
        // Passes the layer information to AddedLayer
        onDrop(layer);
      }
    },
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
      handlerId: monitor.getHandlerId(),
    }),
  }));
  const opacity = isDragging ? 0.4 : 1;

  const HtmlTooltip = styled(({ className, ...props }) => (
    <Tooltip {...props} classes={{ popper: className }} />
  ))(({ theme }) => ({
    [`& .${tooltipClasses.tooltip}`]: {
      backgroundColor: "rgba(255, 255, 255, 0.95)",
      color: "rgba(0, 0, 0, 0.87)",
      maxWidth: 220,
      fontSize: theme.typography.pxToRem(12),
      border: "none",
    },
  }));

  return (
    <RectContainer
      ref2={drag}
      style={{ backgroundColor: COLORS.addLayer, width: 180}}
      dataTestid={`box`}
    >
      <div>
        <HtmlTooltip
          title={
            <React.Fragment>
              <Typography color="inherit">{layer.display_name}</Typography>
              {layer.tooltip_info}
            </React.Fragment>
          }
        >
          <button style={styles.top_left_tooltip}>
            <InfoIcon style={{color: COLORS.layer, fontSize: 18}}/>
          </button>
        </HtmlTooltip>
      </div>
      <p style={{ ...styles.text, opacity }}>{layer.display_name}</p>
    </RectContainer>
  );
};

LayerChoice.propTypes = {
  layer: PropTypes.object.isRequired,
  onDrop: PropTypes.func,
};

export default LayerChoice;

const styles = {
  top_left_tooltip: {
    position: "absolute",
    top: 0,
    left: 0,
    backgroundColor: "transparent",
    borderWidth: 0,
  },
  text: {
    ...GENERAL_STYLES.p,
    color: COLORS.layer,
    fontSize: 25,
  },
};
