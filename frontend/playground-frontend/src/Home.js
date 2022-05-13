import React from "react";
import PropTypes from "prop-types";
import { COLORS, GENERAL_STYLES } from "./constants";
import {
  BackgroundLayout,
  DnD,
  AddedLayer,
  RectContainer,
  AddNewLayer,
  LayerChoice,
} from "./components";

const Home = (props) => {
  return (
    <div style={{ padding: 20 }}>
      <h1 style={styles.h1}>Deep Learning Playground</h1>

      <BackgroundLayout>
        <RectContainer style={{ backgroundColor: COLORS.input }} />
        <AddedLayer />
        <AddedLayer />
        <AddNewLayer />
      </BackgroundLayout>

      <div style={{ marginTop: 20 }} />

      <BackgroundLayout>
        <LayerChoice text="Linear" />
        <LayerChoice text="Non-linear" />
        <LayerChoice text="Other Linear" />
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
