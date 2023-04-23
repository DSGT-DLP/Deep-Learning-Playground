import React from "react";
import { TabularData } from "@/features/Train/features/Tabular/types/tabularTypes";
import { useGetColumnsFromDatasetQuery } from "@/features/Train/redux/trainspaceApi";
import { useAppSelector } from "@/common/redux/hooks";
import { Typography } from "@mui/material";

const TabularParametersStep = ({
  renderStepperButtons,
}: {
  renderStepperButtons: (
    submitTrainspace: (data: TabularData) => void
  ) => React.ReactNode;
}) => {
  const trainspace = useAppSelector(
    (state) => state.trainspace.current as TabularData<"PARAMETERS"> | undefined
  );
  if (!trainspace) return <></>;
  const { data, refetch } = useGetColumnsFromDatasetQuery({
    dataSource: "TABULAR",
    dataset: trainspace.datasetData,
  });
  return <Typography>{data?.toString()}</Typography>;
};

export default TabularParametersStep;
