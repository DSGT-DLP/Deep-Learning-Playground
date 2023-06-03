import React, { useEffect } from "react";
import { TrainspaceData } from "../types/tabularTypes";
import { Stack, Typography } from "@mui/material";
import { useAppDispatch, useAppSelector } from "@/common/redux/hooks";
import { updateTabularTrainspaceData } from "../redux/tabularActions";
import { ReviewData } from "../types/tabularTypes";
import { useForm } from "react-hook-form";
import { TextField } from "@mui/material";
import { Controller } from "react-hook-form";
import { GENERAL_STYLES } from "@/constants";

const TabularReviewStep = ({
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
  const {
    handleSubmit,
    formState: { errors, isDirty },
    control,
  } = useForm<ReviewData>();
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
      <Controller
        control={control}
        name="notificationPhoneNumber"
        rules={{
          // ref: https://stackoverflow.com/a/16699507/11031425 
          pattern: /^(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$/,
        }}
        render={({ field: { onChange } }) => (
          <TextField
            label="Phone Number"
            onChange={onChange}
            error={!!errors.notificationPhoneNumber}
          />
        )}
      />
      {!!errors.notificationPhoneNumber && (
        <p style={GENERAL_STYLES.error_text}>
          Please enter a valid phone number
        </p>
      )}
      <Controller
        control={control}
        name="notificationEmail"
        rules={{ pattern: /^\S+@\S+\.\S+$/ }}
        render={({ field: { onChange } }) => (
          <TextField
            label="Email Address"
            onChange={onChange}
            error={!!errors.notificationEmail}
          />
        )}
      />
      {!!errors.notificationEmail && (
        <p style={GENERAL_STYLES.error_text}>Please enter a valid email</p>
      )}
      {renderStepperButtons((trainspaceData) => {
        handleSubmit((data) => {
          dispatch(
            updateTabularTrainspaceData({
              current: {
                ...trainspaceData,
                reviewData: data,
              },
              stepLabel: "REVIEW",
            })
          );
        })();
      })}
    </Stack>
  );
};

export default TabularReviewStep;
