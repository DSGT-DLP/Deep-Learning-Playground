import DataTable from "react-data-table-component";
import ONNX_OUTPUT_PATH from "../../backend_outputs/my_deep_learning_model.onnx";
import PT_PATH from "../../backend_outputs/model.pt";
import PKL_PATH from "../../backend_outputs/model.pkl";
import Plot from "react-plotly.js";
import React from "react";
import { CSVLink } from "react-csv";
import { GENERAL_STYLES, COLORS, ROUTE_DICT } from "../../constants";
import { DLResultsType, TrainResultsJSONResponseType } from "./TrainButton";
import { Data, Layout, XAxisName, YAxisName } from "plotly.js";

interface ResultsPropsType {
  dlpBackendResponse: TrainResultsJSONResponseType;
  problemType: { label: string; value: string };
  choice: keyof typeof ROUTE_DICT;
  simplified?: boolean;
}
const Results = (props: ResultsPropsType) => {
  const { dlpBackendResponse, problemType, choice } = props;
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
  const auc_roc_data: Data[] = [];
  const category_list_auc =
    dlpBackendResponse?.auxiliary_outputs?.category_list;
  const image_data = dlpBackendResponse?.auxiliary_outputs?.image_data || "";

  auc_roc_data.push({
    name: "baseline",
    x: [0, 1],
    y: [0, 1],
    type: "scatter",
    marker: { color: "grey" },
    line: {
      dash: "dash",
    },
  });
  for (let i = 0; i < auc_roc_data_res.length; i++) {
    auc_roc_data.push({
      name: `${category_list_auc[i]} (AUC: ${auc_roc_data_res[i][2]})`,
      x: auc_roc_data_res[i][0] || [],
      y: auc_roc_data_res[i][1] || [],
      type: "scatter",
    });
  }

  const dl_results_columns_react_csv = Object.keys(
    dl_results_data[0] || []
  ).map((c) => ({
    label: c,
    key: c,
  }));

  const mapResponses = (key: string) =>
    dlpBackendResponse?.dl_results?.map((e) =>
      hasKey(e, key) ? e[key] : ""
    ) || [];

  const FIGURE_HEIGHT = 500;
  const FIGURE_WIDTH = 750;

  const TrainVTestAccuracy = () => {
    if (choice === "classicalml") {
      return null;
    } else if (problemType.value === "classification") {
      return null;
    } else {
      return (
        <Plot
          data={[
            {
              name: "Train accuracy",
              x: mapResponses("epoch"),
              y: mapResponses("train_acc"),
              type: "scatter",
              mode: "markers",
              marker: { color: "red", size: 10 },
            },
            {
              name: "Test accuracy",
              x: mapResponses("epoch"),
              y: mapResponses("val/test acc"),
              type: "scatter",
              mode: "markers",
              marker: { color: "blue", size: 10 },
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
          config={{ responsive: true }}
        />
      );
    }
  };

  const TrainVTestLoss = () => {
    if (choice === "classicalml") {
      return null;
    } else {
      return (
        <Plot
          data={[
            {
              name: "Train loss",
              x: mapResponses("epoch"),
              y: mapResponses("train_loss"),
              type: "scatter",
              mode: "markers",
              marker: { color: "red", size: 10 },
            },
            {
              name: "Test loss",
              x: mapResponses("epoch"),
              y: mapResponses("test_loss"),
              type: "scatter",
              mode: "markers",
              marker: { color: "blue", size: 10 },
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
          config={{ responsive: true }}
        />
      );
    }
  };

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
      config={{ responsive: true }}
    />
  );

  const ConfusionMatrix = () => {
    const cm_data = dlpBackendResponse?.auxiliary_outputs?.confusion_matrix;
    const category_list = dlpBackendResponse?.auxiliary_outputs?.category_list;
    const numerical_category_list =
      dlpBackendResponse?.auxiliary_outputs?.numerical_category_list;

    if (!cm_data?.length) return null;

    const layout: Partial<Layout> = {
      title: "Confusion Matrix (Last Epoch)",
      xaxis: {
        title: "Predicted",
        ticktext: category_list,
        tickvals: numerical_category_list,
      },
      yaxis: {
        title: "Actual",
        autorange: "reversed",
        ticktext: category_list,
        tickvals: numerical_category_list,
        tickangle: 270,
      },
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
          xref: "x1" as XAxisName,
          yref: "y1" as YAxisName,
          x: j,
          y: (i + ROWS - 1) % ROWS,
          text: currentValue.toString(),
          font: {
            color: currentValue > 0 ? "white" : "black",
          },
          showarrow: false,
        };
        if (layout.annotations) {
          layout.annotations.push(result);
        }
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
      {choice === "objectdetection" ? (
        <img src={`data:image/jpeg;base64,${image_data}`} />
      ) : null}

      {choice === "classicalml" ? (
        <span style={{ marginLeft: 8 }}>
          <a href={PKL_PATH} download style={styles.download_csv_res}>
            📄 Download model.pkl File
          </a>
        </span>
      ) : (
        <CSVLink data={dl_results_data} headers={dl_results_columns_react_csv}>
          <button style={{ ...styles.download_csv_res, padding: 5.5 }}>
            📄 Download Results (CSV)
          </button>
          {choice === "objectdetection" ? null : (
            <div>
              <span style={{ marginLeft: 8 }}>
                <a
                  href={ONNX_OUTPUT_PATH}
                  download
                  style={styles.download_csv_res}
                >
                  📄 Download ONNX Output File
                </a>
              </span>
              <span style={{ marginLeft: 8 }}>
                <a href={PT_PATH} download style={styles.download_csv_res}>
                  📄 Download model.pt File
                </a>
              </span>
            </div>
          )}
        </CSVLink>
      )}

      {choice === "classicalml" ? null : (
        <DataTable
          pagination
          highlightOnHover
          columns={Object.keys(dl_results_data[0] || []).map((c) => ({
            name: c,
            selector: (row: DLResultsType) => (hasKey(row, c) ? row[c] : ""),
          }))}
          data={dl_results_data}
        />
      )}

      {choice === "objectdetection" ? null : (
        <div style={{ marginTop: 8 }}>
          <TrainVTestAccuracy />
          <TrainVTestLoss />
          {problemType.value === "classification" &&
          auc_roc_data_res.length !== 0 &&
          !props.simplified ? (
            <AUC_ROC_curves />
          ) : null}
          {problemType.value === "classification" &&
          auc_roc_data_res.length === 0 &&
          !props.simplified ? (
            <p style={{ textAlign: "center" }}>
              No AUC/ROC curve could be generated. If this is not intended,
              check that shuffle is set to true to produce a more balanced
              train/test split which would enable correct AUC score calculation
            </p>
          ) : null}
          {problemType.value === "classification" && !props.simplified ? (
            <ConfusionMatrix />
          ) : null}
        </div>
      )}
    </>
  );
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

function hasKey<O extends object>(obj: O, key: PropertyKey): key is keyof O {
  return key in obj;
}
