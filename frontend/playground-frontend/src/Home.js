import React, { useState, useEffect } from "react";
import { COLORS, GENERAL_STYLES, LAYOUT } from "./constants";
import {
  BOOL_OPTIONS,
  DEFAULT_DATASETS,
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
  CSVInput,
  TrainButton,
} from "./components";
import DSGTLogo from "./images/logos/dsgt-logo-light.png";
import { CRITERIONS } from "./settings";
import { DndProvider } from "react-dnd";
import { HTML5Backend } from "react-dnd-html5-backend";
import DataTable from "react-data-table-component";
import LOSS_VIZ from "./visualization_output/my_loss_plot.png";
import ACC_VIZ from "./visualization_output/my_accuracy_plot.png";

const _TitleText = (props) => {
  const { text } = props;
  return <p style={styles.titleText}>{text}</p>;
};

const Home = () => {
  const [csvData, setCSVData] = useState([]);
  const [csvColumns, setCSVColumns] = useState([]);
  const [dl_results_data, set_dl_results_data] = useState([]);
  const [dlpBackendResponse, setDLPBackendResponse] = useState("");

  // input responses
  const [fileURL, setFileURL] = useState("");
  const [addedLayers, setAddedLayers] = useState([]);
  const [targetCol, setTargetCol] = useState();
  const [features, setFeatures] = useState([]);
  const [problemType, setProblemType] = useState();
  const [criterion, setCriterion] = useState();
  const [optimizerName, setOptimizerName] = useState();
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
  };

  const inputColumnOptions = csvColumns.map((e, i) => ({
    label: e.name,
    value: i,
  }));

  const handleChange = (e) => {
    setTargetCol(e);
    const csvColumnsCopy = JSON.parse(JSON.stringify(inputColumnOptions));
    csvColumnsCopy.splice(e["value"], 1);
    setInputFeatureColumnOptions(csvColumnsCopy);
  };

  async function handleURL(url) {
    const headers = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "Origin",
    };
    try {
      if (url !== "") {
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
          if (headers && row.length == headers.length) {
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
        }

        // prepare columns list from headers
        const columns = headers.map((c) => ({
          name: c,
          selector: (row) => row[c],
        }));

        setCSVData(list);
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
      onChange: handleChange,
    },
    {
      queryText: "Features",
      options: inputFeatureColumnOptions,
      onChange: setFeatures,
      isMultiSelect: true,
    },
    {
      queryText: "Problem Type",
      options: PROBLEM_TYPES,
      onChange: setProblemType,
    },
    {
      queryText: "Optimizer Name",
      options: OPTIMIZER_NAMES,
      onChange: setOptimizerName,
    },
    {
      queryText: "Criterion",
      options: CRITERIONS,
      onChange: setCriterion,
    },
    {
      queryText: "Default",
      options: DEFAULT_DATASETS,
      onChange: setUsingDefaultDataset,
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
    if (dlpBackendResponse?.success) {
      return (
        <>
          <DataTable
            pagination
            highlightOnHover
            columns={csvDataToColumns(dl_results_data)}
            data={dl_results_data}
          />
          <>
            {problemType.value === "classification" ? (
              <img
                src={ACC_VIZ}
                alt="Test accuracy for your Deep Learning Model"
              />
            ) : undefined}
            <img
              src={LOSS_VIZ}
              alt="Train vs. Test for your Deep Learning Model"
            />
          </>
        </>
      );
    } else {
      return (
        dlpBackendResponse.message || (
          <p style={{ textAlign: "left" }}>There are no records to display</p>
        )
      );
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h1 style={styles.h1}>
        <img src={DSGTLogo} alt="DSGT Logo" width="60" height="60" />
        Deep Learning Playground
      </h1>

      <DndProvider backend={HTML5Backend}>
        <_TitleText text="Implemented Layers" />
        <BackgroundLayout>
          <RectContainer style={styles.fileInput}>
            <CSVInput
              data={csvData}
              columns={csvColumns}
              setData={setCSVData}
              setColumns={setCSVColumns}
            />
            <input
              style={{ width: "100%" }}
              placeholder="Or type in URL..."
              value={fileURL}
              onChange={(e) => {
                handleURL(e.target.value);
              }}
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
            set_dl_results_data={set_dl_results_data}
            csvData={csvData}
            setDLPBackendResponse={setDLPBackendResponse}
            fileURL={fileURL}
          />
        </BackgroundLayout>

        <div style={{ marginTop: 20 }} />

        <_TitleText text="Layers Inventory" />

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

      <_TitleText text="Deep Learning Parameters" />

      <BackgroundLayout>
        {input_queries.map((e) => (
          <Input {...e} key={e.queryText} />
        ))}
      </BackgroundLayout>

      <_TitleText text="CSV Input" />

      <DataTable
        pagination
        highlightOnHover
        columns={csvColumns}
        data={csvData}
      />

      <_TitleText text="Deep Learning Results" />

      {showResults()}
    </div>
  );
};

export default Home;

const deepCopyObj = (obj) => JSON.parse(JSON.stringify(obj));

const styles = {
  h1: {
    ...GENERAL_STYLES.p,
    padding: 0,
    margin: "0 0 20px 0",
    display: "flex",
    alignItems: "center",
  },
  titleText: { ...GENERAL_STYLES.p, color: COLORS.layer, fontSize: 20 },
  fileInput: {
    backgroundColor: COLORS.input,
    width: 200,
    ...LAYOUT.column,
    // justifyContent: "space-between",
  },
};

const csvDataToColumns = (data) => {
  if (!data?.length) return;
  const headers = Object.keys(data[0]);
  return headers.map((c) => ({
    name: c,
    selector: (row) => row[c],
  }));
};
