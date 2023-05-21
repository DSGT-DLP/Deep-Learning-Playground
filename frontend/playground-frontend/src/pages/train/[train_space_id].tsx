import Footer from "@/common/components/Footer";
import NavbarMain from "@/common/components/NavBarMain";
import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";
import Paper from "@mui/material/Paper";
import dynamic from "next/dynamic";
import { useRouter } from "next/router";
import { Data, XAxisName, YAxisName } from "plotly.js";
import React from "react";
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const TrainSpace = () => {
  const { train_space_id } = useRouter().query;
  const data = {
    success: true,
    message: "Dataset trained and results outputted successfully",
    dl_results: [
      {
        epoch: 1,
        train_time: 0.029964923858642578,
        train_loss: 1.1126993695894878,
        test_loss: 1.1082043647766113,
        train_acc: 0.3333333333333333,
        "val/test acc": 0.3,
      },
      {
        epoch: 2,
        train_time: 0.0221712589263916,
        train_loss: 1.1002190907796223,
        test_loss: 1.100191593170166,
        train_acc: 0.3333333333333333,
        "val/test acc": 0.3,
      },
      {
        epoch: 3,
        train_time: 0.0680840015411377,
        train_loss: 1.0896958708763123,
        test_loss: 1.0933666229248047,
        train_acc: 0.3333333333333333,
        "val/test acc": 0.3,
      },
      {
        epoch: 4,
        train_time: 0.007375478744506836,
        train_loss: 1.0802951455116272,
        test_loss: 1.0868618488311768,
        train_acc: 0.3333333333333333,
        "val/test acc": 0.3,
      },
      {
        epoch: 5,
        train_time: 0.008754491806030273,
        train_loss: 1.071365197499593,
        test_loss: 1.080164909362793,
        train_acc: 0.3333333333333333,
        "val/test acc": 0.3,
      },
    ],
    auxiliary_outputs: {
      confusion_matrix: [
        [0, 0, 6],
        [0, 0, 8],
        [0, 0, 6],
      ],
      AUC_ROC_curve_data: [
        [
          [0.0, 0.0, 0.0, 0.07142857142857142, 0.07142857142857142, 1.0],
          [
            0.0, 0.16666666666666666, 0.8333333333333334, 0.8333333333333334,
            1.0, 1.0,
          ],
          0.9880952380952381,
        ],
        [
          [
            0.0, 0.08333333333333333, 0.5, 0.5, 0.5833333333333334,
            0.5833333333333334, 0.6666666666666666, 0.6666666666666666, 1.0,
          ],
          [0.0, 0.0, 0.0, 0.75, 0.75, 0.875, 0.875, 1.0, 1.0],
          0.46875,
        ],
        [
          [0.0, 0.0, 0.0, 0.07142857142857142, 0.07142857142857142, 1.0],
          [
            0.0, 0.16666666666666666, 0.8333333333333334, 0.8333333333333334,
            1.0, 1.0,
          ],
          0.9880952380952381,
        ],
      ],
    },
    status: 200,
  };
  return (
    <div style={{ height: "100vh" }}>
      <NavbarMain />
      <Container>
        <h1>{train_space_id}</h1>
        <Grid container spacing={2}>
          <Grid item>
            <Paper>
              <Plot
                data={[
                  {
                    name: "Train accuracy",
                    x: data.dl_results.map((x) => x.epoch),
                    y: data.dl_results.map((x) => x["train_acc"]),
                    type: "scatter",
                    mode: "markers",
                    marker: { color: "red", size: 10 },
                  },
                  {
                    name: "Test accuracy",
                    x: data.dl_results.map((x) => x.epoch),
                    y: data.dl_results.map((x) => x["val/test acc"]),
                    type: "scatter",
                    mode: "markers",
                    marker: { color: "blue", size: 10 },
                  },
                ]}
                layout={{
                  height: 350,
                  width: 525,
                  xaxis: { title: "Epoch Number" },
                  yaxis: { title: "Accuracy" },
                  title: "Train vs. Test Accuracy for your Deep Learning Model",
                  showlegend: true,
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                }}
                config={{ responsive: true }}
              />
            </Paper>
          </Grid>
          <Grid item>
            <Paper>
              <Plot
                data={[
                  {
                    name: "Train loss",
                    x: data.dl_results.map((x) => x.epoch),
                    y: data.dl_results.map((x) => x.train_loss),
                    type: "scatter",
                    mode: "markers",
                    marker: { color: "red", size: 10 },
                  },
                  {
                    name: "Test loss",
                    x: data.dl_results.map((x) => x.epoch),
                    y: data.dl_results.map((x) => x.test_loss),
                    type: "scatter",
                    mode: "markers",
                    marker: { color: "blue", size: 10 },
                  },
                ]}
                layout={{
                  height: 350,
                  width: 525,
                  xaxis: { title: "Epoch Number" },
                  yaxis: { title: "Loss" },
                  title: "Train vs. Test Loss for your Deep Learning Model",
                  showlegend: true,
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                }}
                config={{ responsive: true }}
              />
            </Paper>
          </Grid>
          <Grid item>
            <Paper>
              <Plot
                data={[
                  {
                    name: "baseline",
                    x: [0, 1],
                    y: [0, 1],
                    type: "scatter",
                    marker: { color: "grey" },
                    line: {
                      dash: "dash",
                    },
                  },
                  ...(data.auxiliary_outputs.AUC_ROC_curve_data.map((x) => ({
                    name: `(AUC: ${x[2]})`,
                    x: x[0] as number[],
                    y: x[1] as number[],
                    type: "scatter",
                  })) as Data[]),
                ]}
                layout={{
                  height: 350,
                  width: 525,
                  xaxis: { title: "False Positive Rate" },
                  yaxis: { title: "True Positive Rate" },
                  title: "AUC/ROC Curves for your Deep Learning Model",
                  showlegend: true,
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                }}
                config={{ responsive: true }}
              />
            </Paper>
          </Grid>
          <Grid item>
            <Paper>
              <Plot
                data={[
                  {
                    z: data.auxiliary_outputs.confusion_matrix,
                    type: "heatmap",
                    colorscale: [
                      [0, "#e6f6fe"],
                      [1, "#003058"],
                    ],
                  },
                ]}
                layout={{
                  height: 525,
                  width: 525,
                  title: "Confusion Matrix (Last Epoch)",
                  xaxis: {
                    title: "Predicted",
                  },
                  yaxis: {
                    title: "Actual",
                    autorange: "reversed",
                  },
                  showlegend: true,
                  annotations: data.auxiliary_outputs.confusion_matrix
                    .map((row, i) =>
                      row.map((_, j) => ({
                        xref: "x1" as XAxisName,
                        yref: "y1" as YAxisName,
                        x: j,
                        y:
                          (i +
                            data.auxiliary_outputs.confusion_matrix.length -
                            1) %
                          data.auxiliary_outputs.confusion_matrix.length,
                        text: data.auxiliary_outputs.confusion_matrix[
                          (i +
                            data.auxiliary_outputs.confusion_matrix.length -
                            1) %
                            data.auxiliary_outputs.confusion_matrix.length
                        ][j].toString(),
                        font: {
                          color:
                            data.auxiliary_outputs.confusion_matrix[
                              (i +
                                data.auxiliary_outputs.confusion_matrix.length -
                                1) %
                                data.auxiliary_outputs.confusion_matrix.length
                            ][j] > 0
                              ? "white"
                              : "black",
                        },
                        showarrow: false,
                      }))
                    )
                    .flat(),
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                }}
              />
            </Paper>
          </Grid>
        </Grid>
      </Container>
      <Footer />
    </div>
  );
};

export default TrainSpace;
