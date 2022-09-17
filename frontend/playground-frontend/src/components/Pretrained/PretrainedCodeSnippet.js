import React from "react";
import PropTypes from "prop-types";
import { FaCopy } from "react-icons/fa";
import { toast } from "react-toastify";

const PretrainedCodeSnippet = (props) => {
  const { backendResponse, modelName, n_epochs } = props;

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
  if (modelName) {
    if (modelName.module === "pytorch") {
      return `import torch\nfrom fastai.data.core import DataLoaders\nfrom fastai.vision.all import *\nfrom fastai.vision import *\n\ndevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")\ndls = DataLoaders.from_dsets(train_dataset, valid_dataset, device=device)\nmodel = torchvision.models.${modelName.value}\nlearn = vision_learner(dls, model, n_out=10, loss_func=torch.nn.CrossEntropyLoss(), pretrained=True)\n\nlearn.fit(${n_epochs})`;
    } else if (modelName.module === "timm") {
      return "using timm";
    } else {
      return "unexpected error";
    }
  }
}

PretrainedCodeSnippet.propTypes = {
  backendResponse: PropTypes.shape({
    success: PropTypes.bool,
    message: PropTypes.string,
  }),
  modelName: PropTypes.any,
  n_epochs: PropTypes.number.isRequired,
};

export default PretrainedCodeSnippet;
