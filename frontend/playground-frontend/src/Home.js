import React, { useState } from "react";
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
  const [addedLayers, setAddedLayers] = useState(["Linear"]);

  return (
    <div style={{ padding: 20 }}>
      <h1 style={styles.h1}>Deep Learning Playground</h1>

      <DndProvider backend={HTML5Backend}>
        <BackgroundLayout>
          <RectContainer style={{ backgroundColor: COLORS.input }} />
          {addedLayers.map((e, i) => (
            <AddedLayer
              text={e}
              key={i}
              onDelete={() => {
                const currentLayers = [...addedLayers];
                currentLayers.splice(i);
                setAddedLayers(currentLayers);
              }}
            />
          ))}
          <AddNewLayer />
        </BackgroundLayout>

        <div style={{ marginTop: 20 }} />

        <BackgroundLayout>
          {POSSIBLE_LAYERS.map((e) => (
            <LayerChoice
              text={e}
              key={e}
              onDrop={(newLayerName) => {
                setAddedLayers((currentAddedLayers) => {
                  const copyCurrent = [...currentAddedLayers];
                  copyCurrent.push(newLayerName);
                  return copyCurrent;
                });
              }}
            />
          ))}
        </BackgroundLayout>
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

const POSSIBLE_LAYERS = ["Linear", "Non-linear", "Other linear"];
