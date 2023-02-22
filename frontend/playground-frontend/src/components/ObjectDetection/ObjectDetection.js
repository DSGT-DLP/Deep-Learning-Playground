import React, { useState, useMemo } from "react";
import {
  OBJECT_DETECTION_PROBLEM_TYPES,
  DETECTION_TYPES,
  DETECTION_TRANSFORMS,
} from "../../settings";
import { DEFAULT_DETECTION_TRANSFORMS, COLORS } from "../../constants";
import { DndProvider } from "react-dnd";
import { HTML5Backend } from "react-dnd-html5-backend";
import { FormControlLabel, Switch } from "@mui/material";
import {
  Input,
  TitleText,
  BackgroundLayout,
  AddedLayer,
  AddNewLayer,
  LayerChoice,
  TrainButton,
  Results,
  ChoiceTab,
  Spacer,
  CustomModelName,
} from "../index";
import FilerobotImageEditor, {
  TABS,
  TOOLS,
} from "react-filerobot-image-editor";

const ObjectDetection = () => {
  const [customModelName, setCustomModelName] = useState(
    `Model ${new Date().toLocaleString()}`
  );
  const [problemType, setProblemType] = useState("");
  const [detectionType, setDetectionType] = useState("");
  const [dlpBackendResponse, setDLPBackendResponse] = useState();
  const [beginnerMode, setBeginnerMode] = useState(true);
  const [inputKey, setInputKey] = useState(0);
  const [uploadFile, setUploadFile] = useState(null);
  const [imageTransforms, setImageTransforms] = useState(
    DEFAULT_DETECTION_TRANSFORMS
  );

  const input_responses = {
    problemType: problemType?.value,
    detectionType: detectionType?.value,
    transforms: imageTransforms,
    uploadFile: uploadFile,
  };

  const input_queries = [
    {
      queryText: "ProblemType",
      options: OBJECT_DETECTION_PROBLEM_TYPES,
      onChange: setProblemType,
      defaultValue: problemType,
      beginnerMode: detectionType?.value !== "rekognition" ? true : false,
    },
    {
      queryText: "DetectionType",
      options: DETECTION_TYPES,
      onChange: setDetectionType,
      defaultValue: detectionType,
    },
  ];

  const ResultMemo = useMemo(
    () => (
      <Results
        dlpBackendResponse={dlpBackendResponse}
        problemType={OBJECT_DETECTION_PROBLEM_TYPES[0]}
        choice="objectdetection"
      />
    ),
    [dlpBackendResponse, OBJECT_DETECTION_PROBLEM_TYPES[0]]
  );

  const onClick = () => {
    setBeginnerMode(!beginnerMode);
    setInputKey((e) => e + 1);
  };
  //https://stackoverflow.com/questions/35940290/how-to-convert-base64-string-to-javascript-file-object-like-as-from-file-input-f
  const dataURLtoFile = (dataurl, filename) => {
    const arr = dataurl.split(",");
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n) {
      u8arr[n - 1] = bstr.charCodeAt(n - 1);
      n -= 1; // to make eslint happy
    }
    return new File([u8arr], filename, { type: mime });
  };

  return (
    <div id="image-models">
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
      </DndProvider>
      <BackgroundLayout>
        <TrainButton
          {...input_responses}
          setDLPBackendResponse={setDLPBackendResponse}
          choice="objectdetection"
        />
      </BackgroundLayout>
      <Spacer height={40} />
      <TitleText text="Detection Parameters" />
      <BackgroundLayout>
        {input_queries.map((e) => (
          <Input {...e} key={e.queryText + inputKey} />
        ))}
      </BackgroundLayout>
      <FilerobotImageEditor
        source="https://scaleflex.airstore.io/demo/stephen-walker-unsplash.jpg"
        onSave={(editedImageObject) => {
          const file = dataURLtoFile(
            editedImageObject.imageBase64,
            editedImageObject.fullName
          );
          setUploadFile(file);
        }}
        annotationsCommon={{
          fill: "#ff0000",
        }}
        Text={{ text: "Filerobot..." }}
        Rotate={{ angle: 90, componentType: "slider" }}
        Crop={{
          presetsItems: [
            {
              titleKey: "classicTv",
              descriptionKey: "4:3",
              ratio: 4 / 3,
            },
            {
              titleKey: "cinemascope",
              descriptionKey: "21:9",
              ratio: 21 / 9,
            },
          ],
          presetsFolders: [
            {
              titleKey: "socialMedia", // will be translated into Social Media as backend contains this translation key
              // icon: Social, // optional, Social is a React Function component. Possible (React Function component, string or HTML Element)
              groups: [
                {
                  titleKey: "facebook",
                  items: [
                    {
                      titleKey: "profile",
                      width: 180,
                      height: 180,
                      descriptionKey: "fbProfileSize",
                    },
                    {
                      titleKey: "coverPhoto",
                      width: 820,
                      height: 312,
                      descriptionKey: "fbCoverPhotoSize",
                    },
                  ],
                },
              ],
            },
          ],
        }}
        tabsIds={[TABS.ADJUST, TABS.ANNOTATE, TABS.WATERMARK]} // or {['Adjust', 'Annotate', 'Watermark']}
        defaultTabId={TABS.ANNOTATE} // or 'Annotate'
        defaultToolId={TOOLS.TEXT} // or 'Text'
      />
      <Spacer height={40} />
      <DndProvider backend={HTML5Backend}>
        <TitleText text="Image Transformations" />
        <BackgroundLayout>
          {imageTransforms.map((_, i) => (
            <AddedLayer
              thisLayerIndex={i}
              addedLayers={imageTransforms}
              setAddedLayers={setImageTransforms}
              key={i}
              onDelete={() => {
                const currentLayers = [...imageTransforms];
                currentLayers.splice(i, 1);
                setImageTransforms(currentLayers);
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
        </BackgroundLayout>
        <Spacer height={40} />
        <BackgroundLayout>
          {DETECTION_TRANSFORMS.map((e) => (
            <LayerChoice
              layer={e}
              key={e.display_name}
              onDrop={(newLayer) => {
                setImageTransforms((currentAddedLayers) => {
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
      <TitleText text="Detection Results" />
      {ResultMemo}
    </div>
  );
};

export default ObjectDetection;
const deepCopyObj = (obj) => JSON.parse(JSON.stringify(obj));
