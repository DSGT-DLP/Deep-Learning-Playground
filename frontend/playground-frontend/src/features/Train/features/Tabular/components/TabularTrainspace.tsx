import { useAppSelector } from "@/common/redux/hooks";
import { TrainspaceData } from "@/features/Train/features/Tabular/types/tabularTypes";
import { UseFormHandleSubmit, useForm } from "react-hook-form";
import React from "react";
import TrainspaceLayout from "@/features/Train/components/TrainspaceLayout";
import {
  Button,
  Stack,
  Step,
  StepButton,
  Stepper,
  TextField,
} from "@mui/material";
import {
  TRAINSPACE_SETTINGS,
  STEP_SETTINGS,
} from "../constants/tabularConstants";

const TabularTrainspace = () => {
  const trainspace = useAppSelector(
    (state) => state.trainspace.current as TrainspaceData | undefined
  );
  if (!trainspace) return <></>;
  const {
    handleSubmit,
    formState: { errors },
    register,
  } = useForm<TrainspaceData>({
    defaultValues: trainspace,
  });
  return (
    <TrainspaceLayout
      code={`#Code to train your model locally\nprint("Hello World")`}
      nameField={
        <TextField
          id="filled-basic"
          label="Name"
          variant="outlined"
          required
          helperText={errors.name ? "Name is required" : ""}
          error={errors.name ? true : false}
          fullWidth
          {...register("name", { required: true })}
        />
      }
      stepper={
        <Stepper activeStep={trainspace.step}>
          {TRAINSPACE_SETTINGS.steps.map((step) => (
            <Step key={step}>
              <StepButton>{STEP_SETTINGS[step].name}</StepButton>
            </Step>
          ))}
        </Stepper>
      }
      trainspaceStep={
        <TrainspaceStepInner
          trainspace={trainspace}
          handleSubmit={handleSubmit}
        />
      }
    />
  );
};

const TrainspaceStepInner = ({
  trainspace,
  handleSubmit,
}: {
  trainspace: TrainspaceData;
  handleSubmit: UseFormHandleSubmit<TrainspaceData>;
}) => {
  const Component =
    STEP_SETTINGS[TRAINSPACE_SETTINGS.steps[trainspace.step]].component;

  return (
    <Component
      renderStepperButtons={(submitTrainspace) => (
        <Stack direction={"row"} justifyContent={"space-between"}>
          <Button variant="outlined">Previous</Button>
          <Button
            variant="contained"
            onClick={() => handleSubmit(submitTrainspace)()}
          >
            Next
          </Button>
        </Stack>
      )}
    />
  );
};

export default TabularTrainspace;
