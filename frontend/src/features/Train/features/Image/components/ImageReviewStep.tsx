import React, { useEffect } from "react";
import { TrainspaceData } from "../types/imageTypes";
import { Stack, Typography } from "@mui/material";
import { useAppDispatch, useAppSelector } from "@/common/redux/hooks";
import { updateImageTrainspaceData } from "../redux/imageActions";

const ImageReviewStep = ({
  renderStepperButtons,
  setIsModified,
}: {
  renderStepperButtons: (
    submitTrainspace: (data: TrainspaceData<"REVIEW">) => void
  ) => React.ReactNode;
  setIsModified: React.Dispatch<React.SetStateAction<boolean>>;
}) => {
  const trainspace = useAppSelector(
    (state) => state.trainspace.current as TrainspaceData<"REVIEW"> | undefined
  );
  const dispatch = useAppDispatch();
  if (!trainspace) return <></>;
  useEffect(() => {
    setIsModified(true);
  }, []);
  return (
    <Stack spacing={3}>
      <Typography>{`Dataset: ${trainspace.datasetData.name}`}</Typography>
      <Typography>{`Is Default: ${trainspace.datasetData.isDefaultDataset}`}</Typography>
      <Typography>{`Criterion: ${trainspace.parameterData.criterion}`}</Typography>
      <Typography>{`Optimizer: ${trainspace.parameterData.optimizerName}`}</Typography>
      {renderStepperButtons((trainspaceData) => {
        dispatch(
          updateImageTrainspaceData({
            current: {
              ...trainspaceData,
              reviewData: {},
            },
            stepLabel: "REVIEW",
          })
        );
      })}
    </Stack>
  );
};

export default ImageReviewStep;
