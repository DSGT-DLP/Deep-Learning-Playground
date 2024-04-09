import Footer from "@/common/components/Footer";
import NavbarMain from "@/common/components/NavBarMain";
import { useAppSelector } from "@/common/redux/hooks";
import { isSignedIn } from "@/common/redux/userLogin";
import { useGetTrainResultsDataQuery } from "@/features/Train/redux/trainspaceApi";
import { DetailedTrainResultsData } from "@/features/Train/types/trainTypes";
import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";
import Paper from "@mui/material/Paper";
import dynamic from "next/dynamic";
import { useRouter } from "next/router";
import { Data, XAxisName, YAxisName } from "plotly.js";
import React, { useEffect } from "react";
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const LINE_CHART_COLORS = ["red", "blue", "green"];

const mapTrainResultsDataToCharts = (
  detailedTrainResultsData: DetailedTrainResultsData
) => {
  // sort by graph_index asc and ignore negative graph indices
  const sortedData = detailedTrainResultsData.all_metrics
    .filter((metric) => metric.graph_index >= 0)
    .sort((a, b) => a.graph_index - b.graph_index);
  const charts = [];
  let i = 0;
  while (i < sortedData.length) {
    const metric = sortedData[i];
    if (metric.chart_type === "LINE") {
      const data = [];
      for (let i = 0; i < metric.time_series.length; i++) {
        const time_series = metric.time_series[i];
        data.push({
          name: time_series.y_name,
          x: time_series.x_values,
          y: time_series.y_values,
          type: "scatter",
          mode: "markers",
          marker: { color: LINE_CHART_COLORS[i], size: 10 },
        });
      }
      charts.push(
        <Plot
          data={data as Data[]}
          layout={{
            height: 350,
            width: 525,
            xaxis: { title: metric.time_series[0].x_name },
            // yaxis: { title: "Y axis" },
            title: metric.name,
            showlegend: true,
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
          }}
          config={{ responsive: true }}
        />
      );
    } else if (metric.chart_type === "AUC/ROC") {
      charts.push(
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
            ...(metric.values.map((x) => ({
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
      );
    } else if (metric.chart_type === "CONFUSION_MATRIX") {
      charts.push(
        <Plot
          data={[
            {
              z: metric.values,
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
            annotations: metric.values
              .map((row, i) =>
                row.map((_, j) => ({
                  xref: "x1" as XAxisName,
                  yref: "y1" as YAxisName,
                  x: j,
                  y: (i + metric.values.length - 1) % metric.values.length,
                  text: metric.values[
                    (i + metric.values.length - 1) % metric.values.length
                  ][j].toString(),
                  font: {
                    color:
                      metric.values[
                        (i + metric.values.length - 1) % metric.values.length
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
      );
    } else {
      throw Error("Undefined chart type received");
    }
    i += 1;
  }

  return charts;
};

const TrainSpace = () => {
  const { train_space_id } = useRouter().query;
  const { data, isLoading, refetch, error } = useGetTrainResultsDataQuery({
    trainspaceId: train_space_id,
  });

  const user = useAppSelector((state) => state.currentUser.user);
  const router = useRouter();
  useEffect(() => {
    if (router.isReady && !user) {
      router.replace({ pathname: "/login" });
    }
  }, [user, router.isReady]);
  
  if (error) {
    setTimeout(() => refetch(), 3000);
  }

  if (!isSignedIn(user) || !data || isLoading) {
    return <></>;
  }

  const charts = mapTrainResultsDataToCharts(data);
  return (
    <div style={{ height: "100vh" }}>
      <NavbarMain />
      <Container>
        <h1>{train_space_id}</h1>
        <Grid container spacing={2}>
          {charts.map((chart) => (
            <Grid item>
              <Paper>{chart}</Paper>
            </Grid>
          ))}
        </Grid>
      </Container>
      <Footer />
    </div>
  );
};

export default TrainSpace;
