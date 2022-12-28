import React, { useState } from "react";
import {
  IMAGE_CLASSIFICATION_CRITERION,
  BOOL_OPTIONS,
  OPTIMIZER_NAMES,
  IMAGE_DEFAULT_DATASETS,
  PRETRAINED_MODELS,
  POSSIBLE_TRANSFORMS,
  PROBLEM_TYPES,
  ADV_PRETRAINED_MODELS,
} from "../../settings";
import { DEFAULT_TRANSFORMS } from "../../constants";

import Input from "../Home/Input";
// import RectContainer from "../Home/RectContainer";
import TitleText from "../general/TitleText";
import BackgroundLayout from "../Home/BackgroundLayout";
import Transforms from "../ImageModels/Transforms";
import TrainButton from "../Home/TrainButton";
import ChoiceTab from "../Home/ChoiceTab";
import EmailInput from "../Home/EmailInput";
import Results from "../Home/Results";
import LargeFileUpload from "../general/LargeFileUpload";
import DataCodeSnippet from "../ImageModels/DataCodeSnippet";
import { FormControlLabel, Switch } from "@mui/material";
import CustomModelName from "../Home/CustomModelName";
import Spacer from "../general/Spacer";
// import PretrainedCodeSnippet from "./PretrainedCodeSnippet";
import PytorchCodeSnippet from "./PytorchCodeSnippet";

const Pretrained = () => {
  const [customModelName, setCustomModelName] = useState(
    `Model ${new Date().toLocaleString()}`
  );
  const [dlpBackendResponse, setDLPBackendResponse] = useState();
  const [modelName, setModelName] = useState("");
  const [email, setEmail] = useState("");
  const [criterion, setCriterion] = useState(IMAGE_CLASSIFICATION_CRITERION[0]);
  const [optimizerName, setOptimizerName] = useState(OPTIMIZER_NAMES[0]);
  const [usingDefaultDataset, setUsingDefaultDataset] = useState("");
  const [shuffle, setShuffle] = useState(BOOL_OPTIONS[1]);
  const [epochs, setEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(20);
  const [beginnerMode, setBeginnerMode] = useState(true);
  const [inputKey, setInputKey] = useState(0);
  const [trainTransforms, setTrainTransforms] = useState(DEFAULT_TRANSFORMS);
  const [testTransforms, setTestTransforms] = useState(DEFAULT_TRANSFORMS);
  const [uploadFile, setUploadFile] = useState(null);

  const input_responses = {
    modelName: modelName? modelName.value : null,
    criterion: criterion?.value,
    optimizerName: optimizerName?.value,
    usingDefaultDataset: usingDefaultDataset?.value,
    trainTransforms: trainTransforms,
    testTransforms: testTransforms,
    shuffle: shuffle?.value,
    epochs: epochs,
    batchSize: batchSize,
    email: email,
    uploadFile: uploadFile,
    beginnerMode: beginnerMode,
    customModelName: customModelName,
  };

  const input_queries = [
    {
      queryText: "Model Name",
      options: beginnerMode ? PRETRAINED_MODELS : ADV_PRETRAINED_MODELS,
      onChange: setModelName,
      defaultValue: modelName,
    },
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

  const onClick = () => {
    setBeginnerMode(!beginnerMode);
    setInputKey((e) => e + 1);
  };

  return (
    <div style={{ padding: 30 }}>
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
      <TitleText text="Data & Parameters" />

      <div>
        <div>
          <BackgroundLayout>
            <div className="input-container d-flex flex-column align-items-center justify-content-center">
              <LargeFileUpload
                uploadFile={uploadFile}
                setUploadFile={setUploadFile}
              />
            </div>
            <TrainButton
              {...input_responses}
              setDLPBackendResponse={setDLPBackendResponse}
              choice="pretrained"
              style={{
                container: {
                  width: 155,
                },
              }}
            />
          </BackgroundLayout>
        </div>
        <div>
          <BackgroundLayout>
            {input_queries.map((e) => (
              <Input {...e} key={e.queryText + inputKey} />
            ))}
          </BackgroundLayout>
        </div>
      </div>

      <div style={{ marginTop: 20 }} />
      <TitleText text="Image Transformations" />
      <Transforms
        queryText={"Train Transform"}
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
      <TitleText text="Deep Learning Results" />
      <Results
        dlpBackendResponse={dlpBackendResponse}
        problemType={PROBLEM_TYPES[0]}
      />
      <TitleText text="Code Snippet" />
      {/* <PretrainedCodeSnippet backendResponse={dlpBackendResponse} modelName={modelName} n_epochs={epochs}/> */}
      {
        <PytorchCodeSnippet
          backendResponse={dlpBackendResponse}
          modelName={modelName}
        />
      }
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

export default Pretrained;

// const styles = {
//   fileInput: {
//     ...LAYOUT.column,
//     backgroundColor: COLORS.input,
//     width: 155,
//     height: 75,
//   },
// };
