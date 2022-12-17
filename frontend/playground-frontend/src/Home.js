import React, { useState, useMemo, useEffect } from "react";
import { DEFAULT_ADDED_LAYERS } from "./constants";
import {
  BOOL_OPTIONS,
  CRITERIONS,
  DEFAULT_DATASETS,
  OPTIMIZER_NAMES,
  POSSIBLE_LAYERS,
  PROBLEM_TYPES,
} from "./settings";
import {
  AddNewLayer,
  AddedLayer,
  BackgroundLayout,
  CSVInputFile,
  CSVInputURL,
  CodeSnippet,
  EmailInput,
  Input,
  LayerChoice,
  Spacer,
  Results,
  TitleText,
  TrainButton,
  ChoiceTab,
  CustomModelName,
} from "./components";
import DataTable from "react-data-table-component";
import { DndProvider } from "react-dnd";
import { HTML5Backend } from "react-dnd-html5-backend";
import { toast } from "react-toastify";
import { FormControlLabel, Switch } from "@mui/material";
import { sendToBackend } from "./components/helper_functions/TalkWithBackend";

const Home = () => {
  const [csvDataInput, setCSVDataInput] = useState([]);
  const [uploadedColumns, setUploadedColumns] = useState([]);
  const [dlpBackendResponse, setDLPBackendResponse] = useState();
  const [inputKey, setInputKey] = useState(0);

  // input responses
  const [customModelName, setCustomModelName] = useState(
    `Model ${new Date().toLocaleString()}`
  );
  const [fileURL, setFileURL] = useState("");
  const [email, setEmail] = useState("");
  const [addedLayers, setAddedLayers] = useState(DEFAULT_ADDED_LAYERS);
  const [targetCol, setTargetCol] = useState(null);
  const [features, setFeatures] = useState([]);
  const [problemType, setProblemType] = useState(PROBLEM_TYPES[0]);
  const [criterion, setCriterion] = useState(
    problemType === PROBLEM_TYPES[0] ? CRITERIONS[3] : CRITERIONS[0]
  );
  const [optimizerName, setOptimizerName] = useState(OPTIMIZER_NAMES[0]);
  const [usingDefaultDataset, setUsingDefaultDataset] = useState(
    DEFAULT_DATASETS[0]
  );
  const [shuffle, setShuffle] = useState(BOOL_OPTIONS[1]);
  const [epochs, setEpochs] = useState(5);
  const [testSize, setTestSize] = useState(0.2);
  const [batchSize, setBatchSize] = useState(20);
  const [inputFeatureColumnOptions, setInputFeatureColumnOptions] = useState(
    uploadedColumns.map((e, i) => ({
      label: e.name,
      value: i,
    }))
  );
  const [activeColumns, setActiveColumns] = useState([]);
  const [beginnerMode, setBeginnerMode] = useState(true);

  const input_responses = {
    addedLayers: addedLayers,
    targetCol: targetCol?.label,
    features: features?.map((e) => e.label),
    problemType: problemType?.value,
    criterion: criterion?.value,
    optimizerName: optimizerName?.value,
    usingDefaultDataset: usingDefaultDataset?.value,
    shuffle: shuffle?.value,
    epochs: epochs,
    testSize: testSize,
    batchSize: batchSize,
    fileURL: fileURL,
    email: email,
    customModelName: customModelName,
  };

  const columnOptionsArray = activeColumns.map((e, i) => ({
    label: e.name || e,
    value: i,
  }));

  const inputColumnOptions = usingDefaultDataset.value
    ? []
    : columnOptionsArray;

  const handleTargetChange = (e) => {
    setTargetCol(e);
    const csvColumnsCopy = JSON.parse(JSON.stringify(columnOptionsArray));
    let featuresCopy = JSON.parse(JSON.stringify(features));
    csvColumnsCopy.splice(e.value, 1);
    if (featuresCopy) {
      featuresCopy = featuresCopy.filter((item) => item.value !== e.value);
      setInputKey((e) => e + 1);
      setFeatures(featuresCopy);
    }
    setInputFeatureColumnOptions(csvColumnsCopy);
  };

  const onClick = () => {
    setBeginnerMode(!beginnerMode);
    setInputKey((e) => e + 1);
  };

  const input_queries = [
    {
      queryText: "Target Column",
      options: inputColumnOptions,
      onChange: handleTargetChange,
      defaultValue: targetCol,
    },
    {
      queryText: "Features",
      options: inputFeatureColumnOptions,
      onChange: setFeatures,
      isMultiSelect: true,
      defaultValue: features,
    },
    {
      queryText: "Problem Type",
      options: PROBLEM_TYPES,
      onChange: setProblemType,
      defaultValue: problemType,
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
      options: CRITERIONS.filter((crit) =>
        crit.problem_type.includes(problemType.value)
      ),
      onChange: setCriterion,
      defaultValue: criterion,
      beginnerMode: beginnerMode,
    },
    {
      queryText: "Default",
      options: DEFAULT_DATASETS,
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
      queryText: "Test Size",
      range: true,
      onChange: setTestSize,
      defaultValue: testSize,
    },
    {
      queryText: "Batch Size",
      onChange: setBatchSize,
      defaultValue: batchSize,
      freeInputCustomRestrictions: { type: "number", min: 2 },
      beginnerMode: beginnerMode,
    },
  ];

  useEffect(() => {
    setCriterion(
      problemType === PROBLEM_TYPES[0] ? CRITERIONS[3] : CRITERIONS[0]
    );
    setInputKey((e) => e + 1);
  }, [problemType]);

  useEffect(() => {
    (async () => {
      if (usingDefaultDataset.value) {
        const datasetResult = await sendToBackend("defaultDataset", {
          using_default_dataset: usingDefaultDataset.value,
        });

        if (!datasetResult.success) {
          toast.error(datasetResult.message);
        } else {
          setActiveColumns(datasetResult.columns);
        }
      } else {
        setActiveColumns(uploadedColumns);
      }
    })();
  }, [usingDefaultDataset, uploadedColumns]);

  useEffect(() => {
    if (usingDefaultDataset.value) {
      setTargetCol({ label: "target", value: 0 });
      handleTargetChange(columnOptionsArray[columnOptionsArray.length - 1]);
    } else {
      setTargetCol(null);
      setInputFeatureColumnOptions([]);
    }
    setFeatures(null);
    setInputKey((e) => e + 1);
  }, [activeColumns]);

  const Heading = (
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
  );

  const ImplementedLayers = (
    <>
      <TitleText text="Implemented Layers" />
      <BackgroundLayout>
        <div className="input-container d-flex flex-column align-items-center justify-content-center">
          <CSVInputFile
            setData={setCSVDataInput}
            setColumns={setUploadedColumns}
          />
          <Spacer height={12} />
          <CSVInputURL
            fileURL={fileURL}
            setFileURL={setFileURL}
            setCSVColumns={setUploadedColumns}
            setCSVDataInput={setCSVDataInput}
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
          />
        ))}
        <AddNewLayer />

        <TrainButton
          {...input_responses}
          csvDataInput={csvDataInput}
          setDLPBackendResponse={setDLPBackendResponse}
        />
      </BackgroundLayout>
    </>
  );

  const LayersInventory = (
    <>
      <TitleText text="Layers Inventory" />
      <BackgroundLayout>
        {POSSIBLE_LAYERS.map((e) => (
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
    </>
  );

  const InputParameters = (
    <>
      <TitleText text="Deep Learning Parameters" />
      <BackgroundLayout>
        {input_queries.map((e) => (
          <Input {...e} key={e.queryText + inputKey} />
        ))}
      </BackgroundLayout>
    </>
  );

  const InputCSVDisplay = (
    <>
      <TitleText text="CSV Input" />
      <DataTable
        pagination
        highlightOnHover
        columns={uploadedColumns}
        data={csvDataInput}
      />
    </>
  );

  const ResultsMemo = useMemo(
    () => (
      <Results
        dlpBackendResponse={dlpBackendResponse}
        problemType={problemType}
      />
    ),
    [dlpBackendResponse, problemType]
  );

  return (
    <div id="train-tabular-data" className="container-fluid">
      {Heading}

      <Spacer height={40} />

      <DndProvider backend={HTML5Backend}>
        {ImplementedLayers}
        <Spacer height={40} />
        {LayersInventory}
      </DndProvider>

      <Spacer height={40} />
      {InputParameters}

      <Spacer height={40} />

      <TitleText text="Email (optional)" />
      <EmailInput setEmail={setEmail} />

      <Spacer height={40} />
      {InputCSVDisplay}

      <Spacer height={40} />

      <TitleText text="Deep Learning Results" />
      {ResultsMemo}

      <Spacer height={40} />

      <TitleText text="Code Snippet" />
      <CodeSnippet backendResponse={dlpBackendResponse} layers={addedLayers} />
    </div>
  );
};

export default Home;

const deepCopyObj = (obj) => JSON.parse(JSON.stringify(obj));
