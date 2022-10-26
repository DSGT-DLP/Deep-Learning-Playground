import React, { useState, useMemo } from "react";
//import Transforms from "./Transforms";
//import DataCodeSnippet from "./DataCodeSnippet";
//import LargeFileUpload from "../general/LargeFileUpload";
import {
  BOOL_OPTIONS,
  DEFAULT_DATASETS,
  //POSSIBLE_LAYERS,
  //POSSIBLE_TRANSFORMS,
  //IMAGE_LAYERS,
  PROBLEM_TYPES,
  ML_MODELS,
} from "../../settings";
import {
  //DEFAULT_IMG_LAYERS,
  //DEFAULT_TRANSFORMS,
  COLORS,
} from "../../constants";
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
  CodeSnippet,
  ChoiceTab,
  Spacer,
  CustomModelName,
  CSVInputFile,
  CSVInputURL,
} from "../index";

const ClassicalMLModel = () => {
  const [customModelName, setCustomModelName] = useState(
    `Model ${new Date().toLocaleString()}`
  );
  const [addedLayers, setAddedLayers] = useState([]);
  //const [trainTransforms, setTrainTransforms] = useState(DEFAULT_TRANSFORMS);
  //const [testTransforms, setTestTransforms] = useState(DEFAULT_TRANSFORMS);
  const [usingDefaultDataset, setUsingDefaultDataset] = useState(null);
  const [shuffle, setShuffle] = useState(BOOL_OPTIONS[1]);
  const [email, setEmail] = useState("");
  const [dlpBackendResponse, setDLPBackendResponse] = useState();
  const [beginnerMode, setBeginnerMode] = useState(true);
  const [inputKey, setInputKey] = useState(0);
  const [testSize, setTestSize] = useState(0.2);
  const [problemType, setProblemType] = useState(PROBLEM_TYPES[0]);

  //const [uploadFile, setUploadFile] = useState(null);

  const input_responses = {
    shuffle: shuffle?.value,
    problemType: problemType?.value,
    addedLayers: addedLayers,
    usingDefaultDataset: usingDefaultDataset?.value,
    customModelName: customModelName,
  };

  //TODO: modify this list accordingly to capture inputs for Classical ML
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
      />
    ),
    [dlpBackendResponse, PROBLEM_TYPES[0]]
  );

  const onClick = () => {
    setBeginnerMode(!beginnerMode);
    setInputKey((e) => e + 1);
  };
//whats this
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
            {/* <LargeFileUpload
              uploadFile={uploadFile}
              setUploadFile={setUploadFile}
            /> */}
            <CSVInputFile/>
            <CSVInputURL/>
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
      {/* <Transforms
        queryText={"Train Transform"}
        options={POSSIBLE_TRANSFORMS}
        transforms={trainTransforms}
        setTransforms={setTrainTransforms}
      /> */}
      <Spacer height={10} />
      {/* <Transforms
        queryText={"Test Transform"}
        options={POSSIBLE_TRANSFORMS}
        transforms={testTransforms}
        setTransforms={setTestTransforms}
      /> */}

      <Spacer height={40} />
      <TitleText text="Email (optional)" />
      <EmailInput email={email} setEmail={setEmail} />

      <Spacer height={40} />
      <TitleText text="Machine Learning Results" />
      {ResultMemo}

      <Spacer height={40} />
      <TitleText text="Code Snippet" />
      <CodeSnippet backendResponse={dlpBackendResponse} layers={addedLayers} />
      {/* <DataCodeSnippet
        backendResponse={dlpBackendResponse}
        trainTransforms={trainTransforms}
        testTransforms={testTransforms}
        batchSize={batchSize}
        shuffle={shuffle}
        defaultData={usingDefaultDataset}
      /> */}
    </div>
  );
};

export default ClassicalMLModel;

const deepCopyObj = (obj) => JSON.parse(JSON.stringify(obj));
