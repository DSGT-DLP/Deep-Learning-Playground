import React, { useState, useEffect } from "react";
import PropTypes from "prop-types";
import RectContainer from "./RectContainer";
import { COLORS, GENERAL_STYLES } from "../../constants";
import {
  socket,
  sendEmail,
  train_and_output,
} from "../helper_functions/TalkWithBackend";
import { Circle } from "rc-progress";
import { toast } from "react-toastify";

const TrainButton = (props) => {
  const {
    addedLayers,
    targetCol = null,
    features = null,
    problemType,
    criterion,
    optimizerName,
    usingDefaultDataset = null,
    shuffle,
    epochs,
    testSize,
    batchSize,
    setDLPBackendResponse,
    csvDataInput = null,
    fileURL = null,
    email,
  } = props;

  const [pendingResponse, setPendingResponse] = useState(false);
  const [progress, setProgress] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(() => {
    socket.on("trainingProgress", (progressData) => {
      setProgress(Number.parseFloat(progressData));
    });
    socket.on("trainingResult", (resultData) => {
      setResult(resultData);
    });
  }, [socket]);

  const reset = () => {
    setPendingResponse(false);
    setProgress(null);
    setResult(null);
  };

  const make_user_arch = () => {
    // making a user_arch array by including all added layers and their parameters to make something like:
    // ["nn.Linear(4, 10)", "nn.ReLU()", "nn.Linear(10, 3)", "nn.Softmax()"]
    const user_arch = [];
    addedLayers.forEach((addedLayer) => {
      const parameters = addedLayer.parameters;
      let parameter_call_input = "";
      const parameters_to_be_added = Array(Object.keys(parameters).length);
      Object.values(parameters).forEach((v) => {
        parameters_to_be_added[v.index] = v.value;
      });
      parameters_to_be_added.forEach((e) => {
        parameter_call_input += e + ",";
      });
      // removing the last ','
      parameter_call_input = parameter_call_input.slice(0, -1);

      const callback = `${addedLayer.object_name}(${parameter_call_input})`;
      user_arch.push(callback);
    });
    return user_arch;
  };

  const validateInputs = (user_arch) => {
    let alertMessage = "";
    if (!user_arch?.length)
      alertMessage += "At least one layer must be added. ";
    if (!criterion) alertMessage += "A criterion must be specified. ";
    if (!optimizerName) alertMessage += "An optimizer name must be specified. ";
    if (!problemType) alertMessage += "A problem type must be specified. ";
    if (!usingDefaultDataset) {
      if (!targetCol || !features?.length) {
        alertMessage +=
          "Must specify an input file, target, and features if not selecting default dataset. ";
      }
      for (let i = 0; i < features.length; i++) {
        if (targetCol === features[i]) {
          alertMessage +=
            "A column that is selected as the target column cannot also be a feature column. ";
          break;
        }
      }
      if (!csvDataInput && !fileURL) {
        alertMessage +=
          "Must specify an input file either from local storage or from an internet URL. ";
      }
    }

    if (alertMessage.length === 0) return true;

    toast.error(alertMessage);
    return false;
  };

  const onClick = async () => {
    setPendingResponse(true);
    setDLPBackendResponse(undefined);
    setProgress(0);

    const user_arch = make_user_arch();
    if (!validateInputs(user_arch)) {
      setPendingResponse(false);
      setProgress(null);
      return;
    }

    const csvDataStr = JSON.stringify(csvDataInput);
    console.log(testSize);
    train_and_output(
      user_arch,
      criterion,
      optimizerName,
      problemType,
      targetCol,
      features,
      usingDefaultDataset,
      testSize,
      epochs,
      batchSize,
      shuffle,
      csvDataStr,
      fileURL
    );
  };

  useEffect(() => {
    if (result) {
      if (result.success) {
        if (email?.length) {
          sendEmail(email, problemType);
        }
        toast.success("Training successful! Scroll to see results!");
      } else if (result.message) {
        toast.error("Training failed. Check output traceback message");
      } else {
        toast.error("Training failed. Check your inputs");
      }
      setDLPBackendResponse(result);
      reset();
    }
  }, [result]);

  return (
    <>
      <RectContainer
        style={{
          ...styles.container,
          backgroundColor: pendingResponse ? COLORS.disabled : COLORS.dark_blue,
        }}
      >
        <button
          style={{
            ...styles.button,
            cursor: pendingResponse ? "wait" : "pointer",
          }}
          onClick={onClick}
          disabled={pendingResponse}
        >
          Train!
        </button>
      </RectContainer>
      {pendingResponse ? (
        <div style={{ marginLeft: 5, marginTop: 10, width: 90, height: 90 }}>
          <Circle percent={progress} strokeWidth={4} />
        </div>
      ) : null}
    </>
  );
};

TrainButton.propTypes = {
  addedLayers: PropTypes.arrayOf(PropTypes.object).isRequired,
  batchSize: PropTypes.number.isRequired,
  criterion: PropTypes.string.isRequired,
  csvDataInput: PropTypes.array,
  email: PropTypes.string,
  epochs: PropTypes.number.isRequired,
  features: PropTypes.arrayOf(PropTypes.string),
  fileURL: PropTypes.string,
  optimizerName: PropTypes.string.isRequired,
  problemType: PropTypes.string.isRequired,
  setDLPBackendResponse: PropTypes.func.isRequired,
  shuffle: PropTypes.bool.isRequired,
  targetCol: PropTypes.string,
  testSize: PropTypes.number.isRequired,
  usingDefaultDataset: PropTypes.string,
};

export default TrainButton;

const styles = {
  container: {
    padding: 0,
    width: 130,
    height: 80,
  },
  button: {
    backgroundColor: "transparent",
    border: "none",
    cursor: "pointer",
    height: "100%",
    width: "100%",
    ...GENERAL_STYLES.p,
    fontSize: 25,
    color: "white",
  },
};
