import React from "react";
import { TrainResultsJSONResponseType } from "../Home/TrainButton";
import { DefaultDatasetType, ModelLayer } from "../../settings";

interface DataCodeSnippetPropTypes {
  backendResponse: TrainResultsJSONResponseType;
  trainTransforms: ModelLayer[];
  testTransforms: ModelLayer[];
  shuffle: { label: string; value: boolean };
  batchSize: number;
  defaultData: DefaultDatasetType;
}
const DataCodeSnippet = (props: DataCodeSnippetPropTypes) => {
  const { backendResponse } = props;

  if (backendResponse?.success) {
    return (
      <div>
        <p style={{ margin: "2px" }}>Getting dataloaders</p>
        <textarea
          readOnly
          rows={10}
          style={{ width: "100%" }}
          value={codeSnippetFormat(props)}
        />
      </div>
    );
  }
};

/**
 * Depending on layer passed in, this function builds a string with layer's name, and parameters associated to it (if any)
 * @param {ModelLayer} layer
 * @returns string in form of <layer name>(<parameters>)
 */
function layerToString(layer: ModelLayer) {
  let layerToString = layer.object_name + "(";

  if (layer.parameters != null) {
    const params = Object.keys(layer.parameters);
    if (params?.length !== 0) {
      const paramList = new Array(params.length);
      for (let i = 0; i < params.length; i++) {
        const param = params[i];

        if (typeof layer.parameters[param] !== "undefined") {
          paramList[layer.parameters[param].index] =
            layer.parameters[param].value;
        }
      }
      for (let i = 0; i < paramList.length; i++) {
        layerToString += paramList[i];
        layerToString += ",";
      }

      const layerList = layerToString.split("");
      layerList[layerList.length - 1] = "";
      layerToString = layerList.join("");
    }
  }
  layerToString += ")";
  return layerToString;
}

function codeSnippetFormat(props: DataCodeSnippetPropTypes) {
  let codeSnippet;

  const trainTransform = transformsToString(props.trainTransforms);
  const testTransform = transformsToString(props.testTransforms);

  if (props.defaultData) {
    codeSnippet =
      "import torchvision.datasets as datasets\n" +
      "import torchvision.transforms import transforms\n" +
      "from torch.utils.data import DataLoader\n\n" +
      `train_transform = ${trainTransform}\n` +
      `test_transform = ${testTransform}\n` +
      `train_set = datasets.${props.defaultData.value}(root='/path/to/download', train=True, download=True, transform=train_transform)\n` +
      `test_set = datasets.${props.defaultData.value}(root='/path/to/download', train=False, download=True, transform=test_transform)\n` +
      `train_loader = DataLoader(train_set, batch_size=${props.batchSize}, shuffle=${props.shuffle.label}, drop_last=True)\n` +
      `test_loader = DataLoader(test_set, batch_size=${props.batchSize}, shuffle=${props.shuffle.label}, drop_last=True)\n`;
  } else {
    codeSnippet =
      "import torchvision.datasets as datasets\n" +
      "import torchvision.transforms as transforms\n" +
      "from torch.utils.data import DataLoader\n\n" +
      `train_transform = ${trainTransform}\n` +
      `test_transform = ${testTransform}\n\n` +
      "# unzip the folder and change the path to the unzipped individual folders in root\n" +
      "train_set= datasets.ImageFolder(root='/path/to/train/folder', transform=train_transform)\n" +
      "test_set = datasets.ImageFolder(root='/path/to/test/folder', transform=test_transform)\n" +
      `train_loader = DataLoader(train_set, batch_size=${props.batchSize}, shuffle=${props.shuffle.label}, drop_last=True)\n` +
      `test_loader = DataLoader(test_set, batch_size=${props.batchSize}, shuffle=${props.shuffle.label}, drop_last=True)`;
  }

  return codeSnippet;
}

function transformsToString(transform: ModelLayer[]) {
  const prepend = "transforms.Compose(";
  let transformsToString = prepend;
  const resultingList = [];
  for (let i = 0; i < transform.length; i++) {
    resultingList.push(layerToString(transform[i]));
  }
  transformsToString += resultingList.join(",") + ")";
  return transformsToString;
}

export default DataCodeSnippet;
