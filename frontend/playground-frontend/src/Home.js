import React, { useState, useEffect } from "react";
import { COLORS, GENERAL_STYLES, LAYOUT } from "./constants";
import {
  BOOL_OPTIONS,
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

  // input responses
  const [addedLayers, setAddedLayers] = useState([]);
  const [targetCol, setTargetCol] = useState();
  const [features, setFeatures] = useState([]);
  const [problemType, setProblemType] = useState();
  const [criterion, setCriterion] = useState();
  const [optimizerName, setOptimizerName] = useState();
  const [usingDefaultDataset, setUsingDefaultDataset] = useState(
    BOOL_OPTIONS[0]
  );
  const [shuffle, setShuffle] = useState(BOOL_OPTIONS[1]);
  const [epochs, setEpochs] = useState(5);
  const [testSize, setTestSize] = useState(0.2);
  const input_responses = {
    addedLayers,
    targetCol: targetCol?.label,
    features: features?.map((e) => e.label),
    problemType: problemType?.value,
    criterion: criterion?.value,
    optimizerName: optimizerName?.value,
    usingDefaultDataset: usingDefaultDataset.value,
    shuffle: shuffle?.value,
    epochs,
    testSize,
  };

  const inputColumnOptions = csvColumns.map((e, i) => ({
    label: e.name,
    value: i,
  }));
  const input_queries = [
    {
      queryText: "Target Column",
      options: inputColumnOptions,
      onChange: setTargetCol,
    },
    {
      queryText: "Features",
      options: inputColumnOptions,
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
      options: BOOL_OPTIONS,
      onChange: setUsingDefaultDataset,
      defaultValue: usingDefaultDataset,
    },
    {
      queryText: "Epochs",
      freeInputCustomProps: { type: "number", min: 0 },
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
      freeInputCustomProps: { type: "number", min: 0, max: 1, step: 0.1 },
    },
  ];

  return (
    <div style={{ padding: 20 }}>
      <h1 style={styles.h1}>
        <img src={DSGTLogo} alt="DSGT Logo" width="60" height="60" />
        Deep Learning Playground
      </h1>

      <DndProvider backend={HTML5Backend}>
        <_TitleText text="Implemented Layers" />
        <BackgroundLayout>
          <RectContainer style={{ backgroundColor: COLORS.input, width: 200 }}>
            <CSVInput
              data={csvData}
              columns={csvColumns}
              setData={setCSVData}
              setColumns={setCSVColumns}
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

      <DataTable
        pagination
        highlightOnHover
        columns={csvDataToColumns(dl_results_data)}
        data={dl_results_data}
      />

      {dl_results_data?.length ? (
        <>
          <img src={ACC_VIZ} alt="ACC Viz" />
          <img src={LOSS_VIZ} alt="ACC Viz" />
        </>
      ) : undefined}
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
    fontSize: 13,
    borderRadius: 4,
    fontWeight: 700,
    cursor: "pointer",
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
