import { useAppSelector } from "@/common/redux/hooks";
import { TabularData } from "@/features/Train/features/Tabular/types/tabularTypes";
import { useForm } from "react-hook-form";
import React from "react";
import TrainspaceLayout, {
  TrainspaceStep,
} from "@/features/Train/components/TrainspaceLayout";
import {
  Button,
  Stack,
  Step,
  StepButton,
  Stepper,
  TextField,
} from "@mui/material";
import { DATA_SOURCE_SETTINGS } from "@/features/Train/constants/trainConstants";

const TabularTrainspace = () => {
  const trainspace = useAppSelector(
    (state) => state.trainspace.current as TabularData | undefined
  );
  if (!trainspace) return <></>;
  const {
    handleSubmit,
    formState: { errors },
    register,
  } = useForm<TabularData>({
    defaultValues: trainspace as TabularData,
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
        <Stepper
          activeStep={DATA_SOURCE_SETTINGS[trainspace.dataSource].steps.indexOf(
            trainspace.step
          )}
        >
          {DATA_SOURCE_SETTINGS[trainspace.dataSource].steps.map((step) => (
            <Step key={step}>
              <StepButton>
                {
                  DATA_SOURCE_SETTINGS[trainspace.dataSource].stepsSettings[
                    step
                  ].name
                }
              </StepButton>
            </Step>
          ))}
        </Stepper>
      }
      trainspaceStep={
        <TrainspaceStep
          stepComponent={
            DATA_SOURCE_SETTINGS[trainspace.dataSource].stepsSettings[
              trainspace.step
            ].stepComponent
          }
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
      }
    />
  );
};

export default TabularTrainspace;
