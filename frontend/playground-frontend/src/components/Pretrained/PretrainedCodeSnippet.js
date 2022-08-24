import React from "react";
import PropTypes from "prop-types";

const PretrainedCodeSnippet = (props) => {
  const { backendResponse, trainLayers} = props;

  if (!backendResponse?.success) {
    return (
      backendResponse?.message || (
        <p style={{ textAlign: "center" }}>There are no records to display</p>
      )
    );
  }
  return (
    <textarea
      readOnly
      rows="10"
      style={{ width: "100%" }}
      value={codeSnippetFormat(trainLayers)}
    />
  );
};

/**
 * This function returns necessary code skeleton to train data from local terminal
 * @param {layers[]} layers
 * @returns string with correct python syntax to 'train' data
 */
function codeSnippetFormat(layers) {
  const codeSnippet =
    "import torch\n" +
    "import torch.nn as nn \n" +
    "from torch.autograd import Variable\n" +
    "model = eval('torchvision.models.{}'.format(model_name.lower()))]\n" +
    "learner = vision_learner(\n" +
        "\tdls,\n" +
        "\tmodel,\n" +
        "\tlr=lr,\n" +
        "\topt_func=eval(optimizer_name),\n" +
        "\tpretrained=True,\n" +
        "\tcut=cut,\n" +
        "\tnormalize=False,\n" +
        "\tloss_func=loss_func,\n" +
        "\tn_out=n_classes,\n" +
        "\tn_in=chan_in,\n" +
        "\tmetrics=[train_accuaracy, accuracy]\n" +
    ")";
    layersToString(layers);
  return codeSnippet;
}

/**
 * Given an array of layers, this function builds a string with all elements of the array after they have applied
 * the layerToString() function
 * @param {layers[]} layers
 * @returns string in form of 'self.model = nn.Sequential(*[layerToString(layers[0]),... ,layerToString(layers[N-1])])
 */
function layersToString(layers) {
  const prepend = "self.model = nn.Sequential(*[";
  let layersToString = prepend;
  const resultingList = [];
  for (let i = 0; i < layers.length; i++) {
    resultingList.push(layerToString(layers[i]));
  }
  layersToString += resultingList.join(",") + "])";
  return layersToString;
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
            layer.parameters[param].value;
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

PretrainedCodeSnippet.propTypes = {
  backendResponse: PropTypes.shape({
    success: PropTypes.bool,
    message: PropTypes.string,
  }),
  trainLayers: PropTypes.array.isRequired,
  testLayers: PropTypes.array,
};

export default PretrainedCodeSnippet;
