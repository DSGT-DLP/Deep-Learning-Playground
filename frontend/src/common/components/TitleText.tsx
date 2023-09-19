import React from "react";
import { COLORS } from "../../constants";

interface TitleTextProps {
  text: string;
}

const TitleText = (props: TitleTextProps) => {
  const { text } = props;
  return <h2 style={styles.titleText}>{text}</h2>;
};

export default TitleText;

const styles = {
  titleText: { color: COLORS.layer },
};
