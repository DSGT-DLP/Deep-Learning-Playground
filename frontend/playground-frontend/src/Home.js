import React from "react";
import PropTypes from "prop-types";
import { COLORS } from "./constants";
import { BackgroundLayout, DnD, Layer, RectContainer } from "./components";

const Home = (props) => {
  return (
    <div style={{ padding: 20 }}>
      <h1 style={styles.h1}>Deep Learning Playground</h1>

      <BackgroundLayout>
        <RectContainer style={{backgroundColor: COLORS.input}} />
        <Layer />
        <Layer />
      </BackgroundLayout>

      <DnD />
    </div>
  );
};

Home.propTypes = {};

export default Home;

const styles = {
  h1: {
    fontFamily: "Arial, Helvetica, sans-serif",
    padding: 0,
    margin: 0,
  },
};
