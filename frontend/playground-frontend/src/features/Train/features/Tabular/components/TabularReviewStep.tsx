import React, { useEffect } from "react";
import { TrainspaceData } from "../types/tabularTypes";
import { Stack, Typography } from "@mui/material";
import { useAppDispatch, useAppSelector } from "@/common/redux/hooks";
import { updateTabularTrainspaceData } from "../redux/tabularActions";
import TitleText from "@/common/components/TitleText";
import { TextField } from "@mui/material";

const TabularReviewStep = ({
  renderStepperButtons,
  setIsModified,
}: {
  renderStepperButtons: (
    submitTrainspace: (data: TrainspaceData<"REVIEW">) => void
  ) => React.ReactNode;
  setIsModified: React.Dispatch<React.SetStateAction<boolean>>;
}) => {
  const [email, setEmail] = React.useState("");
  const [phoneNumber, setPhoneNumber] = React.useState("");
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
      <Typography>{`Is Default?: ${trainspace.datasetData.isDefaultDataset}`}</Typography>
      <Typography>{`Target Column: ${trainspace.parameterData.targetCol}`}</Typography>
      <Typography>{`Feature Columns: ${trainspace.parameterData.features.join()}`}</Typography>
      <TextField
        label="Phone Number"
        onChange={(e) => setPhoneNumber(e.target.value)}
      />
      <TextField
        label="Email Address"
        onChange={(e) => setEmail(e.target.value)}
      />
      {renderStepperButtons((trainspaceData) => {
        dispatch(
          updateTabularTrainspaceData({
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

export default TabularReviewStep;
