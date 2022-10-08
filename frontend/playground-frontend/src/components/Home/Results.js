import DataTable from "react-data-table-component";
import ONNX_OUTPUT_PATH from "../../backend_outputs/my_deep_learning_model.onnx";
import PT_PATH from "../../backend_outputs/model.pt";
import Plot from "react-plotly.js";
import PropTypes from "prop-types";
import React from "react";
import { CSVLink } from "react-csv";
import { GENERAL_STYLES, COLORS } from "../../constants";

const Results = (props) => {
  const { dlpBackendResponse, problemType } = props;
  const dl_results_data = dlpBackendResponse?.dl_results || [];

  if (!dlpBackendResponse?.success) {
    return (
      dlpBackendResponse?.message || (
        <p style={{ textAlign: "center" }}>There are no records to display</p>
      )
    );
  }

  const auc_roc_data_res =
    dlpBackendResponse?.auxiliary_outputs?.AUC_ROC_curve_data || [];
  const auc_roc_data = [];
  const category_list_auc = dlpBackendResponse?.auxiliary_outputs?.category_list;

  auc_roc_data.push({
    name: "baseline",
    x: [0, 1],
    y: [0, 1],
    type: "line",
    marker: { color: "grey" },
    line: {
      dash: "dash",
    },
    config: { responsive: true },
  });
  for (let i = 0; i < auc_roc_data_res.length; i++) {
    auc_roc_data.push({
      name: `${category_list_auc[i]} (AUC: ${auc_roc_data_res[i][2]})`,
      x: auc_roc_data_res[i][0] || [],
      y: auc_roc_data_res[i][1] || [],
      type: "line",
      config: { responsive: true },
    });
  }

  const dl_results_columns_react_csv = Object.keys(dl_results_data[0]).map(
    (c) => ({
      label: c,
      key: c,
    })
  );

  const mapResponses = (key) =>
    dlpBackendResponse?.dl_results.map((e) => e[key]) || [];

  const FIGURE_HEIGHT = 500;
  const FIGURE_WIDTH = 750;

  const TrainVTestAccuracy = () => (
    <Plot
      data={[
        {
          name: "Train accuracy",
          x: mapResponses("epoch"),
          y: mapResponses("train_acc"),
          type: "scatter",
          mode: "markers",
          marker: { color: "red", size: 10 },
          config: { responsive: true },
        },
        {
          name: "Test accuracy",
          x: mapResponses("epoch"),
          y: mapResponses("val/test acc"),
          type: "scatter",
          mode: "markers",
          marker: { color: "blue", size: 10 },
          config: { responsive: true },
        },
      ]}
      layout={{
        width: FIGURE_WIDTH,
        height: FIGURE_HEIGHT,
        xaxis: { title: "Epoch Number" },
        yaxis: { title: "Accuracy" },
        title: "Train vs. Test Accuracy for your Deep Learning Model",
        showlegend: true,
      }}
    />
  );

  const TrainVTestLoss = () => (
    <Plot
      data={[
        {
          name: "Train loss",
          x: mapResponses("epoch"),
          y: mapResponses("train_loss"),
          type: "scatter",
          mode: "markers",
          marker: { color: "red", size: 10 },
          config: { responsive: true },
        },
        {
          name: "Test loss",
          x: mapResponses("epoch"),
          y: mapResponses("test_loss"),
          type: "scatter",
          mode: "markers",
          marker: { color: "blue", size: 10 },
          config: { responsive: true },
        },
      ]}
      layout={{
        width: FIGURE_WIDTH,
        height: FIGURE_HEIGHT,
        xaxis: { title: "Epoch Number" },
        yaxis: { title: "Loss" },
        title: "Train vs. Test Loss for your Deep Learning Model",
        showlegend: true,
      }}
    />
  );

  const AUC_ROC_curves = () => (
    <Plot
      data={auc_roc_data}
      layout={{
        width: FIGURE_WIDTH,
        height: FIGURE_HEIGHT,
        xaxis: { title: "False Positive Rate" },
        yaxis: { title: "True Positive Rate" },
        title: "AUC/ROC Curves for your Deep Learning Model",
        showlegend: true,
      }}
    />
  );

  const ConfusionMatrix = () => {
    const cm_data = dlpBackendResponse?.auxiliary_outputs?.confusion_matrix;
    const category_list = dlpBackendResponse?.auxiliary_outputs?.category_list;
    const numerical_category_list = dlpBackendResponse?.auxiliary_outputs?.numerical_category_list;
    console.log(cm_data);
    console.log(numerical_category_list);

    if (!cm_data?.length) return null;

    const layout = {
      title: "Confusion Matrix (Last Epoch)",
      xaxis: { title: "Predicted", ticktext: category_list, tickvals: numerical_category_list},
      yaxis: { title: "Actual", autorange: "reversed", ticktext: category_list, tickvals: numerical_category_list, tickangle: 270,
      showarrow: false},
      showlegend: true,
      width: FIGURE_HEIGHT,
      height: FIGURE_HEIGHT,
      annotations: [],
    };

    const ROWS = cm_data.length;
    const COLS = cm_data[0].length;

    for (let i = 0; i < ROWS; i++) {
      for (let j = 0; j < COLS; j++) {
        const currentValue = cm_data[(i + ROWS - 1) % ROWS][j];
        const result = {
          xref: "x1",
          yref: "y1",
          x: j,
          y: (i + ROWS - 1) % ROWS,
          text: currentValue,
          font: {
            color: currentValue > 0 ? "white" : "black",
          },
          showarrow: false,
        };
        layout.annotations.push(result);
      }
    }
    
    return (
      <Plot
        data={[
          {
            z: dlpBackendResponse?.auxiliary_outputs?.confusion_matrix,
            type: "heatmap",
            colorscale: [
              [0, "#e6f6fe"],
              [1, COLORS.dark_blue],
            ],
          },
        ]}
        layout={layout}
      />
    );
  };

  return (
    <>
      <CSVLink data={dl_results_data} headers={dl_results_columns_react_csv}>
        <button style={{ ...styles.download_csv_res, padding: 5.5 }}>
          📄 Download Results (CSV)
        </button>
      </CSVLink>
      <span style={{ marginLeft: 8 }}>
        <a href={ONNX_OUTPUT_PATH} download style={styles.download_csv_res}>
          📄 Download ONNX Output File
        </a>
      </span>
      <span style={{ marginLeft: 8 }}>
        <a href={PT_PATH} download style={styles.download_csv_res}>
          📄 Download model.pt File
        </a>
      </span>

      <DataTable
        pagination
        highlightOnHover
        columns={Object.keys(dl_results_data[0]).map((c) => ({
          name: c,
          selector: (row) => row[c],
        }))}
        data={dl_results_data}
      />

      <div style={{ marginTop: 8 }}>
        {problemType.value === "classification" ? <TrainVTestAccuracy /> : null}
        <TrainVTestLoss />
        {problemType.value === "classification" &&
        auc_roc_data_res.length !== 0 ? (
          <AUC_ROC_curves />
        ) : null}
        {problemType.value === "classification" &&
        auc_roc_data_res.length === 0 ? (
          <p style={{ textAlign: "center" }}>
            No AUC/ROC curve could be generated. If this is not intended, check
            that shuffle is set to true to produce a more balanced train/test
            split which would enable correct AUC score calculation
          </p>
        ) : null}
        {problemType.value === "classification" ? <ConfusionMatrix /> : null}
      </div>
    </>
  );
};

Results.propTypes = {
  dlpBackendResponse: PropTypes.shape({
    dl_results: PropTypes.array,
    auxiliary_outputs: PropTypes.object,
    message: PropTypes.string.isRequired,
    success: PropTypes.bool.isRequired,
  }),
  problemType: PropTypes.objectOf(PropTypes.string).isRequired,
};

export default Results;

const styles = {
  download_csv_res: {
    ...GENERAL_STYLES.p,
    backgroundColor: COLORS.layer,
    border: "none",
    color: "white",
    cursor: "pointer",
    padding: 8,
    textDecoration: "none",
    fontSize: "medium",
  },
};
