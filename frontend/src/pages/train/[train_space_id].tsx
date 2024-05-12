import Footer from "@/common/components/Footer";
import NavbarMain from "@/common/components/NavBarMain";
import { useAppSelector } from "@/common/redux/hooks";
import { isSignedIn } from "@/common/redux/userLogin";
import { useGetTrainspaceQuery } from "@/features/Train/redux/trainspaceApi";
import { DetailedTrainResultsData } from "@/features/Train/types/trainTypes";
import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";
import Paper from "@mui/material/Paper";
import { useRouter } from "next/router";
import React, { useEffect } from "react";
import {
  mapMetricToLinePlot,
  mapMetricToAucRocPlot,
  mapMetricToConfusionMatrixPlot,
} from "./metrics_to_charts";

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
      charts.push(mapMetricToLinePlot(metric));
    } else if (metric.chart_type === "AUC/ROC") {
      charts.push(mapMetricToAucRocPlot(metric));
    } else if (metric.chart_type === "CONFUSION_MATRIX") {
      charts.push(mapMetricToConfusionMatrixPlot(metric));
    } else {
      throw Error("Undefined chart type received");
    }
    i += 1;
  }

  return charts;
};

const TrainSpace = () => {
  const { train_space_id } = useRouter().query;
  const { data, isLoading, refetch, error } = useGetTrainspaceQuery({
    trainspaceId: train_space_id,
    withResults: true
  });

  const user = useAppSelector((state) => state.currentUser.user);
  const router = useRouter();
  useEffect(() => {
    if (router.isReady && !user) {
      console.log("redirect to login");
      router.replace({ pathname: "/login" });
    }
  }, [user, router.isReady]);

  if (error) {
    setTimeout(() => refetch(), 3000);
  }

  if (!isSignedIn(user) || !data || isLoading) {
    return <></>;
  }

  const charts = mapTrainResultsDataToCharts(data.trainspace.detailedTrainResultsData);
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
