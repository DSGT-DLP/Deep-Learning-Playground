import React from "react";
import { TabularData } from "@/features/Train/features/Tabular/types/tabularTypes";
import { useGetColumnsFromDatasetFileQuery } from "@/features/Train/redux/trainspaceApi";
import { useAppSelector } from "@/common/redux/hooks";
import { FileDatasetData } from "@/features/Train/types/trainTypes";
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
  const { data, refetch } = useGetColumnsFromDatasetFileQuery({
    dataSource: "TABULAR",
    filename: (trainspace.datasetData as FileDatasetData).name,
  });
  return <Typography>{data?.toString()}</Typography>;
};

export default TabularParametersStep;
