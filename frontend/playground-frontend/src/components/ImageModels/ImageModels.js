import React, { useState } from "react";
import Transforms from "./Transforms";
import {
  BOOL_OPTIONS,
  CRITERIONS,
  IMAGE_DEFAULT_DATASETS,
  OPTIMIZER_NAMES,
  POSSIBLE_LAYERS,
  POSSIBLE_TRANSFORMS,
  IMAGE_LAYERS,
  IMAGE_CLASSIFICATION_CRITERION,
} from "../../settings";
import {
  DEFAULT_ADDED_LAYERS,
  DEFAULT_TRANSFORMS,
  COLORS,
  LAYOUT,
  GENERAL_STYLES,
} from "../../constants";
import { DndProvider } from "react-dnd";
import { HTML5Backend } from "react-dnd-html5-backend";

import {
  Input,
  TitleText,
  BackgroundLayout,
  RectContainer,
  AddedLayer,
  AddNewLayer,
  TrainButton,
  LayerChoice,
  EmailInput,
} from "..";

const ImageModels = () => {
  const [addedLayers, setAddedLayers] = useState(DEFAULT_ADDED_LAYERS);
  const [trainTransforms, setTrainTransforms] = useState(DEFAULT_TRANSFORMS);
  const [testTransforms, setTestTransforms] = useState(DEFAULT_TRANSFORMS);
  const [criterion, setCriterion] = useState(IMAGE_CLASSIFICATION_CRITERION[0]);
  const [optimizerName, setOptimizerName] = useState(OPTIMIZER_NAMES[0]);
  const [usingDefaultDataset, setUsingDefaultDataset] = useState();
  const [epochs, setEpochs] = useState(5);
  const [shuffle, setShuffle] = useState(BOOL_OPTIONS[1]);
  const [batchSize, setBatchSize] = useState(20);
  const [email, setEmail] = useState("");
  const [dlpBackendResponse, setDLPBackendResponse] = useState();

  const input_responses = {
    batchSize: batchSize,
    criterion: criterion,
    shuffle: shuffle,
    epochs: epochs,
    optimizerName: optimizerName,
    addedLayers: addedLayers,
    usingDefaultDataset: usingDefaultDataset?.value,
    trainTransforms: trainTransforms,
    testTransforms: testTransforms,
  };

  const input_queries = [
    {
      queryText: "Optimizer Name",
      options: OPTIMIZER_NAMES,
      onChange: setOptimizerName,
      defaultValue: optimizerName,
    },
    {
      queryText: "Criterion",
      options: CRITERIONS,
      onChange: setCriterion,
      defaultValue: criterion,
    },
    {
      queryText: "Default",
      options: IMAGE_DEFAULT_DATASETS,
      onChange: setUsingDefaultDataset,
      defaultValue: usingDefaultDataset,
    },
    {
      queryText: "Epochs",
      freeInputCustomRestrictions: { type: "number", min: 0 },
      onChange: setEpochs,
      defaultValue: epochs,
    },
    {
      queryText: "Shuffle",
      options: BOOL_OPTIONS,
      onChange: setShuffle,
      defaultValue: shuffle,
    },

    {
      queryText: "Batch Size",
      onChange: setBatchSize,
      defaultValue: batchSize,
      freeInputCustomRestrictions: { type: "number", min: 2 },
    },
  ];

  const ALL_LAYERS = POSSIBLE_LAYERS.concat(IMAGE_LAYERS);

  // ADD ZIP INPUT FILE CONTAINER INSIDE <RectContainer style={styles.fileInput}>

  return (
    <div style={{ padding: 20 }}>
      <DndProvider backend={HTML5Backend}>
        <TitleText text="Implemented Layers" />
        <BackgroundLayout>
          <RectContainer style={styles.fileInput}></RectContainer>

          {addedLayers.map((_, i) => (
            <AddedLayer
              thisLayerIndex={i}
              addedLayers={addedLayers}
              setAddedLayers={setAddedLayers}
              key={i}
              onDelete={() => {
                const currentLayers = [...addedLayers];
                currentLayers.splice(i, 1);
                setAddedLayers(currentLayers);
              }}
            />
          ))}
          <AddNewLayer />
          <TrainButton
            {...input_responses}
            setDLPBackendResponse={setDLPBackendResponse}
            choice = "image"
          />
        </BackgroundLayout>

        <div style={{ marginTop: 20 }} />

        <TitleText test="Layers Inventory" />
        <BackgroundLayout>
          {ALL_LAYERS.map((e) => (
            <LayerChoice
              layer={e}
              key={e.display_name}
              onDrop={(newLayer) => {
                setAddedLayers((currentAddedLayers) => {
                  const copyCurrent = [...currentAddedLayers];
                  const layerCopy = deepCopyObj(newLayer);
                  Object.values(layerCopy.parameters).forEach((val) => {
                    val["value"] = "";
                  });
                  copyCurrent.push(layerCopy);
                  return copyCurrent;
                });
              }}
            />
          ))}
        </BackgroundLayout>
      </DndProvider>

      <div style={{ marginTop: 20 }} />

      <TitleText text="Deep Learning Parameters" />
      <BackgroundLayout>
        {input_queries.map((e) => (
          <Input {...e} key={e.queryText} />
        ))}
      </BackgroundLayout>

      <div style={{ marginTop: 20 }} />
      <TitleText text="Image Transformations"/>
      <Transforms
        queryText={"Test Transform"}
        options={POSSIBLE_TRANSFORMS}
        transforms={trainTransforms}
        setTransforms={setTrainTransforms}
      />
      <div style={{ marginTop: 10 }} />
      <Transforms
        queryText={"Test Transform"}
        options={POSSIBLE_TRANSFORMS}
        transforms={testTransforms}
        setTransforms={setTestTransforms}
      />
      <TitleText text="Email (optional)" />
      <EmailInput email={email} setEmail={setEmail} />
    </div>
  );
};

export default ImageModels;

const styles = {
  fileInput: {
    ...LAYOUT.column,
    backgroundColor: COLORS.input,
    width: 200,
  },
  transformChoice: {
    top_left_tooltip: {
      position: "absolute",
      top: 0,
      left: 0,
      backgroundColor: "transparent",
      borderWidth: 0,
    },
    text: {
      ...GENERAL_STYLES.p,
      color: COLORS.layer,
      fontSize: 16,
    },
  },
};

const deepCopyObj = (obj) => JSON.parse(JSON.stringify(obj));
