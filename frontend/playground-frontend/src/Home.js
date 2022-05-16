import React, { useState, useEffect } from "react";
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
import DSGTLogo from "./images/logos/dsgt-logo-light.png";

const Home = () => {
  const [addedLayers, setAddedLayers] = useState([]);
  const [data, setData] = useState([{}]);

  // useEffect(() => {
  //   fetch("/members")
  //     .then((res) => res.json())
  //     .then((data) => {
  //       setData(data);
  //       console.log(data);
  //     });
  // }, []);

  return (
    <div style={{ padding: 20 }}>
      <h1 style={styles.h1}>
        <img src={DSGTLogo} alt="DSGT Logo" width="60" height="60" />
        Deep Learning Playground
      </h1>

      <DndProvider backend={HTML5Backend}>
        <BackgroundLayout>
          <RectContainer style={{ backgroundColor: COLORS.input }} />
          {addedLayers.map((e, i) => (
            <AddedLayer
              layer={e}
              key={i}
              onDelete={() => {
                const currentLayers = [...addedLayers];
                currentLayers.splice(i, 1);
                setAddedLayers(currentLayers);
              }}
            />
          ))}
          <AddNewLayer />
        </BackgroundLayout>

        <div style={{ marginTop: 20 }} />

        <BackgroundLayout>
          {POSSIBLE_LAYERS.map((e) => {
            return (
              <LayerChoice
                layer={e}
                key={e.display_name}
                onDrop={(newLayer) => {
                  setAddedLayers((currentAddedLayers) => {
                    const copyCurrent = [...currentAddedLayers];
                    copyCurrent.push(newLayer);
                    return copyCurrent;
                  });
                }}
              />
            );
          })}
        </BackgroundLayout>
      </DndProvider>
    </div>
  );
};

export default Home;

const styles = {
  h1: {
    fontFamily: "Arial, Helvetica, sans-serif",
    padding: 0,
    margin: "0 0 20px 0",
    display: "flex",
    alignItems: "center",
  },
};

const POSSIBLE_LAYERS = [
  {
    display_name: "Linear",
    object_name: "nn.linear",
    parameters: [
      { display_name: "Input size" },
      { display_name: "Output size" },
    ],
  },
  {
    display_name: "ReLU",
    object_name: "nn.ReLU",
    parameters: [],
  },
  {
    display_name: "Softmax",
    object_name: "nn.Softmax",
    parameters: [],
  },
];
