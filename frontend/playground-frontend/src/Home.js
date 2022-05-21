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
} from "./components";
import { DndProvider } from "react-dnd";
import { HTML5Backend } from "react-dnd-html5-backend";
import DSGTLogo from "./images/logos/dsgt-logo-light.png";
import { CRITERIONS } from "./settings";
import { useDropzone } from "react-dropzone";

const _TitleText = (props) => {
  const { text } = props;
  return <p style={styles.titleText}>{text}</p>;
};

const Home = () => {
  const [selectedFile, setSelectedFile] = useState();
  const [isFilePicked, setIsFilePicked] = useState(false);

  const [addedLayers, setAddedLayers] = useState([]);
  const [problemType, setProblemType] = useState();
  const [criterion, setCriterion] = useState();
  const [optimizerName, setOptimizerName] = useState();
  const [usingDefaultDataset, setUsingDefaultDataset] = useState(
    BOOL_OPTIONS[0]
  );
  const [shuffle, setShuffle] = useState(BOOL_OPTIONS[0]);
  const [epochs, setEpochs] = useState(5);
  const [testSize, setTestSize] = useState(0.2);

  const { acceptedFiles, getRootProps, getInputProps } = useDropzone({
    accept: { "text/csv": [".csv"] },
    maxFiles: 1,
  });

  const files = acceptedFiles.map((file) => {
    console.log(file.path);
    return (
      <p key={file.path} style={{ textAlign: "center" }}>
        {file.path} - {file.size} bytes
      </p>
    );
  });

  // useEffect(() => {
  //   fetch("/run", {
  //     method: "POST",
  //     body: JSON.stringify({
  //       user_arch: [
  //         "nn.Linear(4, 10)",
  //         "nn.ReLU()",
  //         "nn.Linear(10, 3)",
  //         "nn.Softmax()",
  //       ],
  //       criterion: "CELOSS",
  //       optimizer_name: "SGD",
  //       problem_type: "classification",
  //       default: true,
  //       epochs: 10,
  //     }),
  //     headers: {
  //       "Content-type": "application/json; charset=UTF-8",
  //     },
  //   })
  //     .then((res) => res.json())
  //     .then((data) => {
  //       console.log(data);
  //     });
  // }, []);

  const changeHandler = (event) => {
    console.log(event.target.files[0]);
    setSelectedFile(event.target.files[0]);
    setIsFilePicked(true);
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
          <RectContainer style={{ backgroundColor: COLORS.input, width: 200 }}>
            <section className="container">
              <div {...getRootProps({ className: "dropzone" })}>
                <input {...getInputProps()} />
                <p style={{ ...GENERAL_STYLES.p, textAlign: "center" }}>
                  Drop/select File
                </p>
              </div>
              {files}
            </section>
          </RectContainer>
          {addedLayers.map((e, i) => (
            <AddedLayer
              layer={e}
              key={i}
              onDelete={() => {
                const currentLayers = [...addedLayers];
                currentLayers.splice(i, 1);
                setAddedLayers(currentLayers);
              }}
            />
          ))}
          <AddNewLayer />
        </BackgroundLayout>

        <div style={{ marginTop: 20 }} />

        <_TitleText text="Layers Inventory" />

        <BackgroundLayout>
          {POSSIBLE_LAYERS.map((e) => {
            return (
              <LayerChoice
                layer={e}
                key={e.display_name}
                onDrop={(newLayer) => {
                  setAddedLayers((currentAddedLayers) => {
                    const copyCurrent = [...currentAddedLayers];
                    copyCurrent.push(newLayer);
                    return copyCurrent;
                  });
                }}
              />
            );
          })}
        </BackgroundLayout>
      </DndProvider>

      <div style={{ marginTop: 20 }} />

      <_TitleText text="Inputs" />

      <BackgroundLayout>
        <Input
          queryText="Problem Type"
          options={PROBLEM_TYPES}
          onChange={setProblemType}
        />
        <Input
          queryText="Optimizer Name"
          options={OPTIMIZER_NAMES}
          onChange={setOptimizerName}
        />
        <Input
          queryText="Criterion"
          options={CRITERIONS}
          onChange={setCriterion}
        />
      </BackgroundLayout>
      <BackgroundLayout>
        <Input
          queryText="Default"
          options={BOOL_OPTIONS}
          onChange={setUsingDefaultDataset}
          defaultValue={usingDefaultDataset}
        />
        <Input
          queryText="Epochs"
          freeInputCustomProps={{ type: "number", min: 0 }}
          onChange={setEpochs}
          defaultValue={epochs}
        />
        <Input
          queryText="Shuffle"
          options={BOOL_OPTIONS}
          onChange={setShuffle}
          defaultValue={shuffle}
        />
      </BackgroundLayout>
      <BackgroundLayout>
        <Input
          queryText="Test Size"
          onChange={setTestSize}
          defaultValue={testSize}
          freeInputCustomProps={{ type: "number", min: 0, max: 1, step: 0.1 }}
        />
      </BackgroundLayout>
    </div>
  );
};

export default Home;

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
