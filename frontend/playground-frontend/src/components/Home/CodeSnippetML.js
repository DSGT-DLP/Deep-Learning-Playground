import React from "react";
import PropTypes from "prop-types";
import { FaCopy } from "react-icons/fa";
import { toast } from "react-toastify";

const CodeSnippetML = (props) => {
  const { backendResponse, layers } = props;
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
        rows="10"
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
 * @param {layers[]} layers
 * @returns string with correct python syntax to 'train' data
 */
function codeSnippetFormat(layers) {
  const codeSnippet =
    "import sklearn\n" +
    "# import pickle\n\n" +
    "model = " + layerToString(layers[0]) + "\n\n" +
    "## un-comment below code if loading model from a .pkl file, replace PATH with the location path of the .pkl file \n" +
    "# with open(PATH, 'rb') as f: \n" +
    "#\t model = pickle.load(f)\n\n"+
    "model.predict(x)\n \n";
  return codeSnippet;
}

/**
 * Depending on layer passed in, this function builds a string with layer's name, and parameters associated to it (if any)
 * @param {layers} layer
 * @returns string in form of <layer name>(<parameters>)
 */
 export function layerToString(layer) {
  let layerToString = layer.object_name + "(";

  if (layer.parameters !== undefined && layer.parameters !== null) {
    const params = Object.keys(layer.parameters);
    // params : [0: "inputSize", 1:"outputSize"]
    if (params !== null && params !== undefined && params.length !== 0) {
      // const paramList= Array{[params.length]}

      const paramList = new Array(params.length);
      for (let i = 0; i < params.length; i++) {
        const param = params[i];
        // param: "inputSize"

        if (typeof layer.parameters[param] !== "undefined") {
          paramList[layer.parameters[param].index] =
          layer.parameters[param].kwarg +layer.parameters[param].value;
        }
      }
      for (let i = 0; i < paramList.length; i++) {
        layerToString += paramList[i];
        layerToString += ",";
      }

      layerToString = layerToString.split("");
      layerToString[layerToString.length - 1] = "";
      layerToString = layerToString.join("");
      // layerToString = layerToString.substring(0, layerToString.length)
    }
  }
  layerToString += ")";
  return layerToString;
}
CodeSnippetML.propTypes = {
  backendResponse: PropTypes.shape({
    success: PropTypes.bool,
    message: PropTypes.string,
  }),
  layers: PropTypes.array.isRequired,
};

export default CodeSnippetML;
