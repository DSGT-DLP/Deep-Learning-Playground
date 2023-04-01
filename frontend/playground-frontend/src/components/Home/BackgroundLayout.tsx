import React from "react";
import { COLORS } from "../../constants";

interface BackgroundLayoutPropTypes {
  children: React.ReactNode;
}
const BackgroundLayout = (props: BackgroundLayoutPropTypes) => {
  return <div style={styles.layoutBackground} data-testid="layoutBackground">{props.children}</div>;
};

export default BackgroundLayout;

const styles = {
  layoutBackground: {
    color: "black",
    backgroundColor: COLORS.background,
    padding: 10,
    marginVertical: 10,
    display: "flex",
    flexDirection: "row",
    flexWrap: "wrap",
  } as React.CSSProperties,
};
