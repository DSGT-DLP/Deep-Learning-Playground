import React from "react";

interface RectContainerPropTypes {
  children: React.ReactNode;
  style: React.CSSProperties;
  className: string;
  containerRef: React.LegacyRef<HTMLDivElement>;
  dataTestId: React.ReactNode;
}
const RectContainer = (props: RectContainerPropTypes) => {
  return (
    <div
      style={props.style}
      ref={props.containerRef}
      className={props.className}
      data-testid={props.dataTestId}
    >
      {props.children}
    </div>
  );
};

export default RectContainer;
