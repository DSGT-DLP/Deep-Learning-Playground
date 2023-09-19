import React from "react";
import IconButton from "@mui/material/IconButton";
import Tooltip, { tooltipClasses } from "@mui/material/Tooltip";
import { styled } from "@mui/material/styles";

interface DlpTooltipProps {
  children: React.ReactNode;
  title: React.ReactNode;
  isIcon?: boolean;
  onClick?: () => void;
  className?: string;
  disabled?: boolean;
}

const DlpTooltip = (props: DlpTooltipProps) => {
  const { children, title, isIcon, onClick, className, disabled } = props;

  if (isIcon) {
    return (
      <Tooltip title={title} className={className}>
        <span>
          <IconButton
            onClick={onClick}
            style={{ width: "fit-content", aspectRatio: 1 }}
            disabled={disabled}
            className={className}
          >
            {children}
          </IconButton>
        </span>
      </Tooltip>
    );
  }

  return styled(
    ({
      className,
      title,
      children,
      ...props
    }: {
      className?: string;
      children: React.ReactElement;
      title: React.ReactNode;
    }) => (
      <Tooltip title={title} {...props} classes={{ popper: className }}>
        {children}
      </Tooltip>
    )
  )(({ theme }) => ({
    [`& .${tooltipClasses.tooltip}`]: {
      backgroundColor: "rgba(255, 255, 255, 0.95)",
      color: "rgba(0, 0, 0, 0.87)",
      maxWidth: 220,
      fontSize: theme.typography.pxToRem(12),
      border: "none",
    },
  }));
};

export default DlpTooltip;
