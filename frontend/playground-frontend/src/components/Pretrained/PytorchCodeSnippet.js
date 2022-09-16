import React from "react";
import PropTypes from "prop-types";
import { FaCopy } from "react-icons/fa";
import { toast } from "react-toastify";

const PytorchCodeSnippet = (props) => {
  const { backendResponse, modelName, n_epochs } = props;

  console.log(backendResponse);

  if (!backendResponse?.success) {
    return (
      backendResponse?.message || (
        <p style={{ textAlign: "center" }}>There are no records to display</p>
      )
    );
  }

  const value = codeSnippetFormat(modelName, n_epochs);
  return (
    <div id="code-snippet-div">
      <textarea id="code-snippet-text" readOnly rows="10" value={value} />
      <button
        id="code-snippet-clipboard"
        onClick={() => {
          navigator.clipboard.writeText(value);
          toast.info("Code snippet copied");
        }}
      >
        <FaCopy />
      </button>
    </div>
  );
};

/**
 * This function returns necessary code skeleton to train data from local terminal
 * @returns string with correct python syntax to 'train' data
 */
function codeSnippetFormat(modelName, n_epochs) {
    console.log(n_epochs);
  if (modelName) {
    return `class GetModel(nn.Module):\n\tdef __init__(self, n_classses, in_chan):\n\t\tsuper().__init__()\n\t\tself.cnn = timm.create_model(${modelName.value}, pretrained = True, num_classes = n_classes, in_chans = in_chan)\n
    \tdef forward(self, x):\n\t\tx = self.cnn(x)\n\t\treturn x`;
  }
}

PytorchCodeSnippet.propTypes = {
  backendResponse: PropTypes.shape({
    success: PropTypes.bool,
    message: PropTypes.string,
  }),
  modelName: PropTypes.any,
  n_epochs: PropTypes.number.isRequired,
};

export default PytorchCodeSnippet;
