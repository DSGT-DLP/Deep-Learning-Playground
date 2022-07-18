import React, { useState } from "react";
import { COLORS, DEFAULT_ADDED_LAYERS, LAYOUT } from "./constants";
import {
  BOOL_OPTIONS,
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
  RectContainer,
  Results,
  TitleText,
  TrainButton,
} from "./components";
import DataTable from "react-data-table-component";
import { CRITERIONS } from "./settings";
import { DndProvider } from "react-dnd";
import { HTML5Backend } from "react-dnd-html5-backend";

const Home = () => {
  const [csvDataInput, setCSVDataInput] = useState([]);
  const [csvColumns, setCSVColumns] = useState([]);
  const [dlpBackendResponse, setDLPBackendResponse] = useState();

  // input responses
  const [fileURL, setFileURL] = useState("");
  const [email, setEmail] = useState("");
  const [addedLayers, setAddedLayers] = useState(DEFAULT_ADDED_LAYERS);
  const [targetCol, setTargetCol] = useState();
  const [features, setFeatures] = useState([]);
  const [problemType, setProblemType] = useState(PROBLEM_TYPES[0]);
  const [criterion, setCriterion] = useState(CRITERIONS[3]);
  const [optimizerName, setOptimizerName] = useState(OPTIMIZER_NAMES[0]);
  const [usingDefaultDataset, setUsingDefaultDataset] = useState();
  const [shuffle, setShuffle] = useState(BOOL_OPTIONS[1]);
  const [epochs, setEpochs] = useState(5);
  const [testSize, setTestSize] = useState(0.2);
  const [inputFeatureColumnOptions, setInputFeatureColumnOptions] = useState(
    csvColumns.map((e, i) => ({
      label: e.name,
      value: i,
    }))
  );
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
    fileURL: fileURL,
    email: email,
  };

  const inputColumnOptions = csvColumns.map((e, i) => ({
    label: e.name,
    value: i,
  }));

  const auc_roc_data_res =
    dlpBackendResponse?.auxiliary_outputs?.AUC_ROC_curve_data || [];
  const auc_roc_data = [];
  auc_roc_data.push({
    name: "baseline",
    x: [0, 1],
    y: [0, 1],
    type: "line",
    marker: { color: "grey" },
    line: {
      dash: "dash",
    },
    config: { responsive: true },
  });
  for (var i = 0; i < auc_roc_data_res.length; i++) {
    auc_roc_data.push({
      name: `${i} (AUC: ${auc_roc_data_res[i][2]})`,
      x: auc_roc_data_res[i][0] || [],
      y: auc_roc_data_res[i][1] || [],
      type: "line",
      config: { responsive: true },
    });
  }

  const handleTargetChange = (e) => {
    setTargetCol(e);
    const csvColumnsCopy = JSON.parse(JSON.stringify(inputColumnOptions));
    csvColumnsCopy.splice(e.value, 1);
    setInputFeatureColumnOptions(csvColumnsCopy);
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
    },
    {
      queryText: "Criterion",
      options: CRITERIONS,
      onChange: setCriterion,
      defaultValue: criterion,
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
    },
    {
      queryText: "Test Size",
      onChange: setTestSize,
      defaultValue: testSize,
      freeInputCustomRestrictions: { type: "number", min: 0, step: 0.1 },
    },
  ];

  return (
    <div style={{ padding: 20 }}>
      <DndProvider backend={HTML5Backend}>
        <TitleText text="Implemented Layers" />
        <BackgroundLayout>
          <RectContainer style={styles.fileInput}>
            <CSVInputFile
              data={csvDataInput}
              columns={csvColumns}
              setData={setCSVDataInput}
              setColumns={setCSVColumns}
            />
            <CSVInputURL
              fileURL={fileURL}
              setFileURL={setFileURL}
              setCSVColumns={setCSVColumns}
              setCSVDataInput={setCSVDataInput}
            />
          </RectContainer>

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

        <div style={{ marginTop: 20 }} />

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

      <TitleText text="Email (optional)" />
      <EmailInput email={email} setEmail={setEmail} />

      <TitleText text="CSV Input" />
      <DataTable
        pagination
        highlightOnHover
        columns={csvColumns}
        data={csvDataInput}
      />

      <TitleText text="Deep Learning Results" />
      <Results
        dlpBackendResponse={dlpBackendResponse}
        problemType={problemType}
        auc_roc_data={auc_roc_data}
        auc_roc_data_res={auc_roc_data_res}
      />

      <TitleText text="Code Snippet" />
      <CodeSnippet backendResponse={dlpBackendResponse} layers={addedLayers} />
    </div>
  );
};

export default Home;

const deepCopyObj = (obj) => JSON.parse(JSON.stringify(obj));

const styles = {
  fileInput: {
    ...LAYOUT.column,
    backgroundColor: COLORS.input,
    width: 200,
  },
};
