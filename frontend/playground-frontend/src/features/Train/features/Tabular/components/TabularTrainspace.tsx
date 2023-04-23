import { useAppSelector } from "@/common/redux/hooks";
import { TabularData } from "@/features/Train/features/Tabular/types/tabularTypes";
import { useForm } from "react-hook-form";
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
import { DATA_SOURCE_SETTINGS } from "@/features/Train/constants/trainConstants";
import { TrainspaceStep } from "@/features/Train/types/trainTypes";

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
        <TabularTrainspaceStep
          step={trainspace.step}
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

const TabularTrainspaceStep = ({
  step,
  renderStepperButtons,
}: {
  step: TrainspaceStep<"TABULAR">;
  renderStepperButtons: (
    submitTrainspace: (data: TabularData) => void
  ) => React.ReactNode;
}) => {
  const StepComponent =
    DATA_SOURCE_SETTINGS["TABULAR"].stepsSettings[step].stepComponent;
  return <StepComponent renderStepperButtons={renderStepperButtons} />;
};

export default TabularTrainspace;
