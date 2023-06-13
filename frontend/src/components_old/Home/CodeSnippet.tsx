import React from "react";
import { FaCopy } from "react-icons/fa";
import { toast } from "react-toastify";
import { TrainResultsJSONResponseType } from "./TrainButton";

interface CodeSnippetPropTypes {
  backendResponse: TrainResultsJSONResponseType;
}
const CodeSnippet = (props: CodeSnippetPropTypes) => {
  const { backendResponse } = props;
  const layers = backendResponse?.auxiliary_outputs.user_arch;
  if (!backendResponse?.success) {
    return (
      backendResponse?.message || (
        <p style={{ textAlign: "center" }}>There are no records to display</p>
      )
    );
  }

  return (
    <div id="code-snippet-div">
      <textarea
        id="code-snippet-text"
        readOnly
        rows={10}
        value={codeSnippetFormat(layers)}
      />
      <button
        id="code-snippet-clipboard"
        onClick={() => {
          navigator.clipboard.writeText(codeSnippetFormat(layers));
          toast.info("Code snippet copied", { autoClose: 1000 });
        }}
      >
        <FaCopy />
      </button>
    </div>
  );
};

/**
 * This function returns necessary code skeleton to train data from local terminal
 * @param {string[]} layers
 * @returns string with correct python syntax to 'train' data
 */
function codeSnippetFormat(layers: string[]) {
  const codeSnippet =
    "import torch\n" +
    "import torch.nn as nn \n" +
    "from torch.autograd import Variable\n" +
    "class DLModel(nn.Module):\n" +
    "\tdef __init__(self):\n" +
    "\t\t" +
    layersToString(layers) +
    "\n" +
    "\t\t## un-comment below code if loading model from a .pt file, replace PATH with the location path of the .pt file \n" +
    "\t\t# self.model = torch.load('PATH') \n" +
    "\t\t# self.model.eval()" +
    "\n \n" +
    "\tdef forward(self, x): \n" +
    "\t\t" +
    "self.model(x)";
  return codeSnippet;
}

/**
 * Given an array of layers, this function builds a string with all elements of the array after they have applied
 * the layerToString() function
 * @param {string[]} layers
 * @returns string in form of 'self.model = nn.Sequential(*[layerToString(layers[0]),... ,layerToString(layers[N-1])])
 */
function layersToString(layers: string[]) {
  const prepend = "self.model = nn.Sequential(*[";
  let layersToString = prepend;
  const resultingList = [];
  for (let i = 0; i < layers.length; i++) {
    resultingList.push(layers[i]);
  }
  layersToString += resultingList.join(",") + "])";
  return layersToString;
}

export default CodeSnippet;
