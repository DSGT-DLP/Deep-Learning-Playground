import React, { useState, useMemo } from "react";
import {
  BOOL_OPTIONS,
  DEFAULT_DATASETS,
  PROBLEM_TYPES,
  ML_MODELS,
} from "../../settings";
import { DndProvider } from "react-dnd";
import { HTML5Backend } from "react-dnd-html5-backend";
import { FormControlLabel, Switch } from "@mui/material";
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
  CodeSnippetML,
  ChoiceTab,
  Spacer,
  CustomModelName,
  CSVInputFile,
  CSVInputURL,
} from "../index";

const ClassicalMLModel = () => {
  const [customModelName, setCustomModelName] = useState(
    `Model ${new Date().toUTCString()}`
  );
  const [addedLayers, setAddedLayers] = useState([]);
  const [usingDefaultDataset, setUsingDefaultDataset] = useState(null);
  const [shuffle, setShuffle] = useState(BOOL_OPTIONS[1]);
  const [email, setEmail] = useState("");
  const [dlpBackendResponse, setDLPBackendResponse] = useState();
  const [beginnerMode, setBeginnerMode] = useState(true);
  const [inputKey, setInputKey] = useState(0);
  const [testSize, setTestSize] = useState(0.2);
  const [problemType, setProblemType] = useState(PROBLEM_TYPES[0]);
  const input_responses = {
    shuffle: shuffle?.value,
    problemType: problemType?.value,
    addedLayers: addedLayers,
    usingDefaultDataset: usingDefaultDataset?.value,
    customModelName: customModelName,
  };

  const input_queries = [
    {
      queryText: "Default",
      options: DEFAULT_DATASETS,
      onChange: setUsingDefaultDataset,
      defaultValue: usingDefaultDataset,
    },
    {
      queryText: "TestSize",
      freeInputCustomRestrictions: { type: "number", min: 0 },
      onChange: setTestSize,
      defaultValue: testSize,
    },
    {
      queryText: "ProblemType",
      options: PROBLEM_TYPES,
      onChange: setProblemType,
      defaultValue: problemType,
    },
    {
      queryText: "Shuffle",
      options: BOOL_OPTIONS,
      onChange: setShuffle,
      defaultValue: shuffle,
      beginnerMode: beginnerMode,
    },
  ];
  const ALL_LAYERS = ML_MODELS;

  const ResultMemo = useMemo(
    () => (
      <Results
        dlpBackendResponse={dlpBackendResponse}
        problemType={PROBLEM_TYPES[0]}
        choice="classicalml"
      />
    ),
    [dlpBackendResponse, PROBLEM_TYPES[0]]
  );

  const onClick = () => {
    setBeginnerMode(!beginnerMode);
    setInputKey((e) => e + 1);
  };

  return (
    <div id="ml-models">
      <DndProvider backend={HTML5Backend}>
        <div className="d-flex flex-row justify-content-between">
          <FormControlLabel
            control={<Switch id="mode-switch" onClick={onClick}></Switch>}
            label={`${beginnerMode ? "Enable" : "Disable"} Advanced Settings`}
          />
          <CustomModelName
            customModelName={customModelName}
            setCustomModelName={setCustomModelName}
          />
          <ChoiceTab />
        </div>

        <Spacer height={40} />
        <TitleText text="Implemented Layers" />
        <BackgroundLayout>
          <div className="input-container d-flex flex-column align-items-center justify-content-center">
            <CSVInputFile />
            <CSVInputURL />
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
            />
          ))}
          {addedLayers.length >= 1 ? null : <AddNewLayer />}
          <TrainButton
            {...input_responses}
            setDLPBackendResponse={setDLPBackendResponse}
            choice="classicalml"
          />
        </BackgroundLayout>

        <Spacer height={40} />

        <TitleText text="Model Inventory" />
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

      <TitleText text="Machine Learning Parameters" />
      <BackgroundLayout>
        {input_queries.map((e) => (
          <Input {...e} key={e.queryText + inputKey} />
        ))}
      </BackgroundLayout>

      <Spacer height={40} />
      <TitleText text="Data Transformations" />
      <Spacer height={10} />

      <Spacer height={40} />
      <TitleText text="Email (optional)" />
      <EmailInput email={email} setEmail={setEmail} />

      <Spacer height={40} />
      <TitleText text="Machine Learning Results" />
      {ResultMemo}

      <Spacer height={40} />
      <TitleText text="Code Snippet" />
      <CodeSnippetML
        backendResponse={dlpBackendResponse}
        layers={addedLayers}
      />
    </div>
  );
};

export default ClassicalMLModel;

const deepCopyObj = (obj) => JSON.parse(JSON.stringify(obj));
