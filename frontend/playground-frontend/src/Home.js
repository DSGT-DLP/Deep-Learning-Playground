import React from "react";
import PropTypes from "prop-types";
import { COLORS, GENERAL_STYLES } from "./constants";
import {
  BackgroundLayout,
  Container,
  AddedLayer,
  RectContainer,
  AddNewLayer,
  LayerChoice,
  AddNewLayer2,
} from "./components";
import { DndProvider } from "react-dnd";
import { HTML5Backend } from "react-dnd-html5-backend";

const Home = (props) => {
  return (
    <div style={{ padding: 20 }}>
      <h1 style={styles.h1}>Deep Learning Playground</h1>

      <DndProvider backend={HTML5Backend}>
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

        <Container />
      </DndProvider>
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
