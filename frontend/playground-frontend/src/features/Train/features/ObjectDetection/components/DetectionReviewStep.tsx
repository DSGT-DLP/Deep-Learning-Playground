import React, { useEffect } from "react";
import { TrainspaceData } from "../types/detectionTypes";
import { Stack, Typography } from "@mui/material";
import { useAppDispatch, useAppSelector } from "@/common/redux/hooks";
import { updateDetectionTrainspaceData } from "../redux/detectionActions";
import { ReviewData } from "../types/detectionTypes";
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
      <Typography>{`Detection Type: ${trainspace.parameterData.detectionType}`}</Typography>
      <Typography>{`Detection Problem Type: ${trainspace.parameterData.detectionProblemType}`}</Typography>
      <Controller
        control={control}
        name="notificationPhoneNumber"
        rules={{
          pattern: /^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$/,
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
            updateDetectionTrainspaceData({
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
