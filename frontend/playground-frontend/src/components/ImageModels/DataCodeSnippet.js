import React from "react";
import { PropTypes } from "prop-types";
import { layerToString } from "../Home/CodeSnippet";

const DataCodeSnippet = (props) => {
  const { backendResponse } = props;
  console.log(backendResponse);

  if (backendResponse?.success) {
    return (
      <div>
        <textarea
          readOnly
          rows="10"
          style={{ width: "100%", marginTop: "10px"}}
          value={codeSnippetFormat(props)}
        />
      </div>
    );
  }
};

function codeSnippetFormat(props) {
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
      `train_loader = DataLoader(train_set, batch_size=${props.batchSize}, shuffle=${props.shuffle.label})\n` +
      `test_loader = DataLoader(test_set, batch_size=${props.batchSize}, shuffle=${props.shuffle.label})\n`;
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
      `train_loader = DataLoader(train_set, batch_size=${props.batchSize}, shuffle=${props.shuffle.label})\n` +
      `test_loader = DataLoader(test_set, batch_size=${props.batchSize}, shuffle=${props.shuffle.label})`;
  }

  return codeSnippet;
}

function transformsToString(transform) {
  const prepend = "transforms.Compose(";
  let transformsToString = prepend;
  const resultingList = [];
  for (let i = 0; i < transform.length; i++) {
    resultingList.push(layerToString(transform[i]));
  }
  transformsToString += resultingList.join(",") + ")";
  return transformsToString;
}

DataCodeSnippet.propTypes = {
  backendResponse: PropTypes.shape({
    success: PropTypes.bool,
    message: PropTypes.string,
  }),
};

export default DataCodeSnippet;
