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
  console.log(layers);
  const codeSnippet =
    create_import_statement(layers[0])+"\n" +
    "# import pickle\n\n" +
    "model = " + layerToString(layers[0]) + "\n\n" +
    "## un-comment below code if loading model from a .pkl file, replace PATH with the location path of the .pkl file \n" +
    "# with open(PATH, 'rb') as f: \n" +
    "#\t model = pickle.load(f)\n\n"+
    "model.predict(x)\n \n";
  return codeSnippet;
}

export function create_import_statement(layer){
  const full_model_name = layer.object_name;
  const components = full_model_name.split(".");
  const model_name = components[components.length-1];
  
  const import_statement = "from " + components.slice(0,components.length-1).join(".") + " import " + model_name;
  return import_statement;
}
/**
 * Depending on layer passed in, this function builds a string with layer's name, and parameters associated to it (if any)
 * @param {layers} layer
 * @returns string in form of <layer name>(<parameters>)
 */
 export function layerToString(layer) {
  const components = layer.object_name.split(".");
  let layerToString = components[components.length-1] + "(";

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
          console.log(layer.parameters[param].parameter_type);
          if (layer.parameters[param].parameter_type === "text"){
            paramList[layer.parameters[param].index] =
            layer.parameters[param].kwarg + "\""+ layer.parameters[param].value + "\"";
          }else{
            paramList[layer.parameters[param].index] =
            layer.parameters[param].kwarg +layer.parameters[param].value;
          }
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
