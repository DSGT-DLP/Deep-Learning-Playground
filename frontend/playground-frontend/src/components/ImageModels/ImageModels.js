import React, { useState, useMemo } from "react";
import Transforms from "./Transforms";
import DataCodeSnippet from "./DataCodeSnippet";
import LargeFileUpload from "../general/LargeFileUpload";
import {
  BOOL_OPTIONS,
  IMAGE_DEFAULT_DATASETS,
  OPTIMIZER_NAMES,
  POSSIBLE_LAYERS,
  POSSIBLE_TRANSFORMS,
  IMAGE_LAYERS,
  IMAGE_CLASSIFICATION_CRITERION,
  PROBLEM_TYPES,
} from "../../settings";
import {
  DEFAULT_IMG_LAYERS,
  DEFAULT_TRANSFORMS,
  COLORS,
} from "../../constants";
import { DndProvider } from "react-dnd";
import { HTML5Backend } from "react-dnd-html5-backend";
import { Switch } from "@mui/material";

import {
  Input,
  TitleText,
  BackgroundLayout,
  AddedLayer,
  AddNewLayer,
  TrainButton,
  LayerChoice,
  EmailInput,
  Results,
  CodeSnippet,
  ChoiceTab,
} from "../index";
import Spacer from "../general/Spacer";

const ImageModels = () => {
  const [addedLayers, setAddedLayers] = useState(DEFAULT_IMG_LAYERS);
  const [trainTransforms, setTrainTransforms] = useState(DEFAULT_TRANSFORMS);
  const [testTransforms, setTestTransforms] = useState(DEFAULT_TRANSFORMS);
  const [criterion, setCriterion] = useState(IMAGE_CLASSIFICATION_CRITERION[0]);
  const [optimizerName, setOptimizerName] = useState(OPTIMIZER_NAMES[0]);
  const [usingDefaultDataset, setUsingDefaultDataset] = useState(null);
  const [epochs, setEpochs] = useState(5);
  const [shuffle, setShuffle] = useState(BOOL_OPTIONS[1]);
  const [batchSize, setBatchSize] = useState(20);
  const [email, setEmail] = useState("");
  const [dlpBackendResponse, setDLPBackendResponse] = useState();
  const [beginnerMode, setBeginnerMode] = useState(true);
  const [inputKey, setInputKey] = useState(0);
  const [uploadFile, setUploadFile] = useState(null);

  const input_responses = {
    batchSize: batchSize,
    criterion: criterion?.value,
    shuffle: shuffle?.value,
    epochs: epochs,
    optimizerName: optimizerName?.value,
    addedLayers: addedLayers,
    usingDefaultDataset: usingDefaultDataset?.value,
    trainTransforms: trainTransforms,
    testTransforms: testTransforms,
    uploadFile: uploadFile,
  };

  const input_queries = [
    {
      queryText: "Optimizer Name",
      options: OPTIMIZER_NAMES,
      onChange: setOptimizerName,
      defaultValue: optimizerName,
      beginnerMode: beginnerMode,
    },
    {
      queryText: "Criterion",
      options: IMAGE_CLASSIFICATION_CRITERION,
      onChange: setCriterion,
      defaultValue: criterion,
      beginnerMode: beginnerMode,
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
      beginnerMode: beginnerMode,
    },

    {
      queryText: "Batch Size",
      onChange: setBatchSize,
      defaultValue: batchSize,
      freeInputCustomRestrictions: { type: "number", min: 2 },
      beginnerMode: beginnerMode,
    },
  ];
  const ALL_LAYERS = POSSIBLE_LAYERS.concat(IMAGE_LAYERS);

  const ResultMemo = useMemo(
    () => (
      <Results
        dlpBackendResponse={dlpBackendResponse}
        problemType={PROBLEM_TYPES[0]}
      />
    ),
    [dlpBackendResponse, PROBLEM_TYPES[0]]
  );

  const onClick = () => {
    setBeginnerMode(!beginnerMode);
    setInputKey((e) => e + 1);
  };

  return (
    <div id="image-models">
      <DndProvider backend={HTML5Backend}>
        <ChoiceTab />
        <div>{beginnerMode ? "Beginner" : "Advanced"}</div>
        <Switch id="mode-switch" onClick={onClick}></Switch>
        <Spacer height={40} />
        <TitleText text="Implemented Layers" />
        <BackgroundLayout>
          <div className="input-container d-flex flex-column align-items-center justify-content-center">
            <LargeFileUpload
              uploadFile={uploadFile}
              setUploadFile={setUploadFile}
            />
          </div>

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
              style={{
                input_box: {
                  margin: 7.5,
                  backgroundColor: "white",
                  width: 170,
                  paddingInline: 5,
                },
                layer_box: {
                  width: 150,
                  backgroundColor: COLORS.layer,
                },
              }}
            />
          ))}
          <AddNewLayer />
          <TrainButton
            {...input_responses}
            setDLPBackendResponse={setDLPBackendResponse}
            choice="image"
          />
        </BackgroundLayout>

        <Spacer height={40} />

        <TitleText text="Layers Inventory" />
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
                    val.value = val.default ? val.default : val.min;
                  });
                  copyCurrent.push(layerCopy);
                  return copyCurrent;
                });
              }}
            />
          ))}
        </BackgroundLayout>
      </DndProvider>

      <Spacer height={40} />

      <TitleText text="Deep Learning Parameters" />
      <BackgroundLayout>
        {input_queries.map((e) => (
          <Input {...e} key={e.queryText + inputKey} />
        ))}
      </BackgroundLayout>

      <Spacer height={40} />
      <TitleText text="Image Transformations" />
      <Transforms
        queryText={"Train Transform"}
        options={POSSIBLE_TRANSFORMS}
        transforms={trainTransforms}
        setTransforms={setTrainTransforms}
      />
      <Spacer height={10} />
      <Transforms
        queryText={"Test Transform"}
        options={POSSIBLE_TRANSFORMS}
        transforms={testTransforms}
        setTransforms={setTestTransforms}
      />

      <Spacer height={40} />
      <TitleText text="Email (optional)" />
      <EmailInput email={email} setEmail={setEmail} />

      <Spacer height={40} />
      <TitleText text="Deep Learning Results" />
      {ResultMemo}

      <Spacer height={40} />
      <TitleText text="Code Snippet" />
      <CodeSnippet backendResponse={dlpBackendResponse} layers={addedLayers} />
      <DataCodeSnippet
        backendResponse={dlpBackendResponse}
        trainTransforms={trainTransforms}
        testTransforms={testTransforms}
        batchSize={batchSize}
        shuffle={shuffle}
        defaultData={usingDefaultDataset}
      />
    </div>
  );
};

export default ImageModels;

const deepCopyObj = (obj) => JSON.parse(JSON.stringify(obj));
