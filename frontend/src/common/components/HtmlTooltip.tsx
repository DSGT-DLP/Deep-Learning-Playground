import { tooltipClasses, Tooltip } from "@mui/material";
import { styled } from "@mui/material/styles";
import React from "react";

const HtmlTooltip = styled(
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

export default HtmlTooltip;
