import React from "react";
import PropTypes from "prop-types";
import { FaCopy } from "react-icons/fa";
import { toast } from "react-toastify";

const CodeSnippetML = (props) => {
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
 * @param {string[]} layers
 * @returns string with correct python syntax to 'train' data
 */
function codeSnippetFormat(layers) {
  const codeSnippet =
    create_import_statement(layers[0]) +
    "\n" +
    "# import pickle\n\n" +
    "model = " +
    layers[0].split(".").at(-1) +
    "\n\n" +
    "## un-comment below code if loading model from a .pkl file, replace PATH with the location path of the .pkl file \n" +
    "# with open(PATH, 'rb') as f: \n" +
    "#\t model = pickle.load(f)\n\n" +
    "model.predict(x)\n \n";
  return codeSnippet;
}

export function create_import_statement(layer) {
  const full_model_name = layer.split("(")[0];
  const components = full_model_name.split(".");
  const model_name = components[components.length - 1];

  const import_statement =
    "from " +
    components.slice(0, components.length - 1).join(".") +
    " import " +
    model_name;
  return import_statement;
}

CodeSnippetML.propTypes = {
  backendResponse: PropTypes.shape({
    auxiliary_outputs: PropTypes.object,
    success: PropTypes.bool,
    message: PropTypes.string,
  }),
  layers: PropTypes.array.isRequired,
};

export default CodeSnippetML;
