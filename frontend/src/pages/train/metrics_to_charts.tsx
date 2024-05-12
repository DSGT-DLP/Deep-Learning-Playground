import {
  AucRocChart,
  ConfusionMatrixChart,
  TimeSeriesChart,
} from "@/features/Train/types/trainTypes";
import dynamic from "next/dynamic";
import { Data, XAxisName, YAxisName } from "plotly.js";
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const LINE_CHART_COLORS = ["red", "blue", "green"];

const mapMetricToLinePlot = (metric: TimeSeriesChart) => {
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
  return (
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
};

const mapMetricToAucRocPlot = (metric: AucRocChart) => {
  return (
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
          x: x[0],
          y: x[1],
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
};

const mapMetricToConfusionMatrixPlot = (metric: ConfusionMatrixChart) => {
  return (
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
};

export {
  mapMetricToLinePlot,
  mapMetricToAucRocPlot,
  mapMetricToConfusionMatrixPlot,
};
