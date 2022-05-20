import React, { useState, useEffect } from "react";
import { COLORS, GENERAL_STYLES } from "./constants";
import {
  DEFAULT_OPTIONS,
  OPTIMIZER_NAMES,
  POSSIBLE_LAYERS,
  PROBLEM_TYPES,
} from "./settings";
import {
  BackgroundLayout,
  Container,
  AddedLayer,
  RectContainer,
  AddNewLayer,
  LayerChoice,
  Input,
} from "./components";
import { DndProvider } from "react-dnd";
import { HTML5Backend } from "react-dnd-html5-backend";
import DSGTLogo from "./images/logos/dsgt-logo-light.png";
import { CRITERIONS } from "./settings";

const _TitleText = (props) => {
  const { text } = props;

  return (
    <p style={{ ...GENERAL_STYLES.p, color: COLORS.layer, fontSize: 20 }}>
      {text}
    </p>
  );
};

const Home = () => {
  const [addedLayers, setAddedLayers] = useState([]);
  const [data, setData] = useState([{}]);
  const [problemType, setProblemType] = useState("");
  const [criterion, setCriterion] = useState("");
  const [optimizerName, setOptimizerName] = useState("");
  const [usingDefaultDataset, setUsingDefaultDataset] = useState(false);
  const [epochs, setEpochs] = useState();

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
        <_TitleText>Implemented Layers</_TitleText>
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

        <_TitleText>Layers Inventory</_TitleText>

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

      <div style={{ marginTop: 20 }} />

      <_TitleText>Inputs</_TitleText>

      <BackgroundLayout>
        <Input
          queryText="Problem Type"
          options={PROBLEM_TYPES}
          onChange={setProblemType}
        />
        <Input
          queryText="Optimizer Name"
          options={OPTIMIZER_NAMES}
          onChange={setOptimizerName}
        />
        <Input
          queryText="Criterion"
          options={CRITERIONS}
          onChange={setCriterion}
        />
      </BackgroundLayout>
      <BackgroundLayout>
        <Input
          queryText="Default"
          options={DEFAULT_OPTIONS}
          onChange={setUsingDefaultDataset}
        />
        <Input queryText="Epochs" inputType="number" onChange={setEpochs} />
      </BackgroundLayout>
    </div>
  );
};

export default Home;

const styles = {
  h1: {
    ...GENERAL_STYLES.p,
    padding: 0,
    margin: "0 0 20px 0",
    display: "flex",
    alignItems: "center",
  },
};
