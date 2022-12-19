import React, { useState, useMemo } from "react";
import ImageFileUpload from "../general/ImageFileUpload";
import {
  PROBLEM_TYPES,
} from "../../settings";
import { DndProvider } from "react-dnd";
import { HTML5Backend } from "react-dnd-html5-backend";
import { FormControlLabel, Switch } from "@mui/material";
import {
  Input,
  TitleText,
  BackgroundLayout,
  TrainButton,
  EmailInput,
  Results,
  ChoiceTab,
  Spacer,
  CustomModelName,
} from "../index";

const ObjectDetection = () => {
  const [customModelName, setCustomModelName] = useState(
    `Model ${new Date().toLocaleString()}`
  );
  const [email, setEmail] = useState("");
  const [problemType, setProblemType] = useState("");
  const [dlpBackendResponse, setDLPBackendResponse] = useState();
  const [beginnerMode, setBeginnerMode] = useState(true);
  const [inputKey, setInputKey] = useState(0);
  const [uploadFile, setUploadFile] = useState(null);

  const input_responses = {
    problemType: problemType?.value,
    uploadFile: uploadFile,
  };

  const input_queries = [
    {
      queryText: "ProblemType",
      options: PROBLEM_TYPES,
      onChange: setProblemType,
      defaultValue: problemType,
    },
  ];

  const ResultMemo = useMemo(
    () => (
      <Results
        dlpBackendResponse={dlpBackendResponse}
        problemType={PROBLEM_TYPES[0]}
        choice="objectdetection"
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
          <ImageFileUpload
              uploadFile={uploadFile}
              setUploadFile={setUploadFile}
            />
          </div>
          <TrainButton
            {...input_responses}
            setDLPBackendResponse={setDLPBackendResponse}
            choice="objectdetection"
          />
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
      <TitleText text="Email (optional)" />
      <EmailInput email={email} setEmail={setEmail} />

      <Spacer height={40} />
      <TitleText text="Machine Learning Results" />
      {ResultMemo}
    </div>
  );
};

export default ObjectDetection;

