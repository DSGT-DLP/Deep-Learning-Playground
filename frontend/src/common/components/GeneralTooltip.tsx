import React from "react";
import IconButton from "@mui/material/IconButton";
import Tooltip from "@mui/material/Tooltip";

interface MTooltipProps {
  el: JSX.Element;
  title: string;
  isIcon?: boolean;
  onClick?: () => void;
  className?: string;
  disabled?: boolean;
}

const MTooltip = (props: MTooltipProps) => {
  const { el, title, isIcon, onClick, className, disabled } = props;
  return (
    <Tooltip title={title} className={className}>
      {isIcon ? (
        <span>
          <IconButton
            onClick={onClick}
            style={{ width: "fit-content", aspectRatio: 1 }}
            disabled={disabled}
            className={className}
          >
            {el}
          </IconButton>
        </span>
      ) : (
        el
      )}
    </Tooltip>
  );
};

export default MTooltip;
