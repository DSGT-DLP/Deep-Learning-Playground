import React, { useState, useEffect } from "react";
import {
  COLORS,
  DEFAULT_ADDED_LAYERS,
  GENERAL_STYLES,
  LAYOUT,
} from "./constants";
import {
  BOOL_OPTIONS,
  DEFAULT_DATASETS,
  OPTIMIZER_NAMES,
  POSSIBLE_LAYERS,
  PROBLEM_TYPES,
} from "./settings";
import {
  BackgroundLayout,
  AddedLayer,
  RectContainer,
  AddNewLayer,
  LayerChoice,
  Input,
  CSVInput,
  TrainButton,
  EmailInput,
  TitleText,
  CodeSnippet
} from "./components";
import { CRITERIONS } from "./settings";
import { DndProvider } from "react-dnd";
import { HTML5Backend } from "react-dnd-html5-backend";
import DataTable from "react-data-table-component";
import CONFUSION_VIZ from "./backend_outputs/visualization_output/my_confusion_matrix.png";
import ONNX_OUTPUT_PATH from "./backend_outputs/my_deep_learning_model.onnx";
import PT_PATH from "./backend_outputs/model.pt";
import { CSVLink } from "react-csv";
import Plot from "react-plotly.js";

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
  const [shuffle, setShuffle] = useState(BOOL_OPTIONS[0]);
  const [epochs, setEpochs] = useState(5);
  const [testSize, setTestSize] = useState(0.2);
  const [inputFeatureColumnOptions, setInputFeatureColumnOptions] = useState(
    csvColumns.map((e, i) => ({
      label: e.name,
      value: i,
    }))
  );
  const input_responses = {
    addedLayers,
    targetCol: targetCol?.label,
    features: features?.map((e) => e.label),
    problemType: problemType?.value,
    criterion: criterion?.value,
    optimizerName: optimizerName?.value,
    usingDefaultDataset: usingDefaultDataset?.value,
    shuffle: shuffle?.value,
    epochs,
    testSize,
    fileURL,
    email,
  };

  const inputColumnOptions = csvColumns.map((e, i) => ({
    label: e.name,
    value: i,
  }));

  const dl_results_data = dlpBackendResponse?.dl_results || [];
  const train_loss_data = dlpBackendResponse?.train_loss_results;

  const handleTargetChange = (e) => {
    setTargetCol(e);
    const csvColumnsCopy = JSON.parse(JSON.stringify(inputColumnOptions));
    csvColumnsCopy.splice(e.value, 1);
    setInputFeatureColumnOptions(csvColumnsCopy);
  };

  async function handleURL(url) {
    const headers = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "Origin",
    };
    try {
      if (!url) return;

      let response = await fetch(url);
      response = await response.text();
      const responseLines = response.split(/\r\n|\n/);
      const headers = responseLines[0].split(
        /,(?![^"]*"(?:(?:[^"]*"){2})*[^"]*$)/
      );

      const list = [];
      for (let i = 1; i < responseLines.length; i++) {
        const row = responseLines[i].split(
          /,(?![^"]*"(?:(?:[^"]*"){2})*[^"]*$)/
        );
        if (headers && row.length === headers.length) {
          const obj = {};
          for (let j = 0; j < headers.length; j++) {
            let d = row[j];
            if (d.length > 0) {
              if (d[0] == '"') d = d.substring(1, d.length - 1);
              if (d[d.length - 1] == '"') d = d.substring(d.length - 2, 1);
            }
            if (headers[j]) {
              obj[headers[j]] = d;
            }
          }

          // remove the blank rows
          if (Object.values(obj).filter((x) => x).length > 0) {
            list.push(obj);
          }
        }

        // prepare columns list from headers
        const columns = headers.map((c) => ({
          name: c,
          selector: (row) => row[c],
        }));

        setCSVDataInput(list);
        setCSVColumns(columns);
        setFileURL(url);
      }
    } catch (e) {
      console.log("Incorrect URL");
    }
  }

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

  const showResults = () => {
    if (!dlpBackendResponse?.success) {
      return (
        dlpBackendResponse?.message || (
          <p style={{ textAlign: "center" }}>There are no records to display</p>
        )
      );
    }

    const dl_results_columns_react_csv = Object.keys(dl_results_data[0]).map(
      (c) => ({
        label: c,
        key: c,
      })
    );

    return (
      <>
        <CSVLink data={dl_results_data} headers={dl_results_columns_react_csv}>
          <button style={styles.download_csv_res}>
            📄 Download Results (CSV)
          </button>
        </CSVLink>

        <DataTable
          pagination
          highlightOnHover
          columns={Object.keys(dl_results_data[0]).map((c) => ({
            name: c,
            selector: (row) => row[c],
          }))}
          data={dl_results_data}
        />
        <span>
          <a href={ONNX_OUTPUT_PATH} download style={styles.download_csv_res}>
            📈 Download ONNX Output File
          </a>
        </span>
        <span style={{ marginLeft: 8 }}>
          <a href={PT_PATH} download style={styles.download_csv_res}>
            📈 Download model.pt File
          </a>
        </span>
        <div style={{ marginTop: 8 }}>
          {problemType.value === "classification" ? (
            <Plot
              data={[
                {
                  name: "Train accuracy",
                  x: train_loss_data?.epochs || [],
                  y: train_loss_data?.train_acc || [],
                  type: "scatter",
                  mode: "markers",
                  marker: { color: "red", size: 10 },
                  config: { responsive: true },
                },
                {
                  name: "Test accuracy",
                  x: train_loss_data?.epochs || [],
                  y: train_loss_data?.test_acc || [],
                  type: "scatter",
                  mode: "markers",
                  marker: { color: "blue", size: 10 },
                  config: { responsive: true },
                },
              ]}
              layout={{
                width: 750,
                height: 750,
                xaxis: { title: "Epoch Number" },
                yaxis: { title: "Accuracy" },
                title: "Train vs. Test Accuracy for your Deep Learning Model",
                showlegend: true,
              }}
            />
          ) : undefined}
          <Plot
            data={[
              {
                name: "Train loss",
                x: train_loss_data?.epochs || [],
                y: train_loss_data?.train_loss || [],
                type: "scatter",
                mode: "markers",
                marker: { color: "red", size: 10 },
                config: { responsive: true },
              },
              {
                name: "Test loss",
                x: train_loss_data?.epochs || [],
                y: train_loss_data?.test_loss || [],
                type: "scatter",
                mode: "markers",
                marker: { color: "blue", size: 10 },
                config: { responsive: true },
              },
            ]}
            layout={{
              width: 750,
              height: 750,
              xaxis: { title: "Epoch Number" },
              yaxis: { title: "Loss" },
              title: "Train vs. Test Loss for your Deep Learning Model",
              showlegend: true,
            }}
          />
          <img src={CONFUSION_VIZ} alt="Confusion matrix for the last epoch of your Deep Learning Model" />
          <a href={CONFUSION_VIZ} download style={styles.download_csv_res}>
          📈 Download Confusion matrix
        </a>
        </div>
      </>
    );
  };

  return (
    <div style={{ padding: 20 }}>
      <DndProvider backend={HTML5Backend}>
        <TitleText text="Implemented Layers" />
        <BackgroundLayout>
          <RectContainer style={styles.fileInput}>
            <CSVInput
              data={csvDataInput}
              columns={csvColumns}
              setData={setCSVDataInput}
              setColumns={setCSVColumns}
            />
            <input
              style={{ width: "100%" }}
              placeholder="Or type in URL..."
              value={fileURL}
              onChange={(e) => setFileURL(e.target.value)}
              onBlur={(e) => handleURL(e.target.value)}
            />
          </RectContainer>

          {addedLayers.map((e, i) => (
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
      <EmailInput email={email} setEmail={setEmail} />
      <TitleText text="CSV Input" />
      <DataTable
        pagination
        highlightOnHover
        columns={csvColumns}
        data={csvDataInput}
      />
      <TitleText text="Deep Learning Results" />
      {showResults()}
      <TitleText text = "Code Snippet"/>
      <CodeSnippet backendResponse={dlpBackendResponse} layers={addedLayers}></CodeSnippet>
      
    </div>
  );
};

export default Home;

const deepCopyObj = (obj) => JSON.parse(JSON.stringify(obj));

const styles = {
  h1: {
    ...GENERAL_STYLES.p,
    padding: 0,
    margin: 0,
    display: "flex",
    alignItems: "center",
  },
  fileInput: {
    backgroundColor: COLORS.input,
    width: 200,
    ...LAYOUT.column,
  },
  download_csv_res: {
    backgroundColor: COLORS.layer,
    textDecoration: "none",
    border: "none",
    color: "white",
    ...GENERAL_STYLES.p,
    padding: 8,
    cursor: "pointer",
  },
};
