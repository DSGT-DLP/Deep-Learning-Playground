import { useAppDispatch, useAppSelector } from "@/common/redux/hooks";
import { TrainspaceData } from "@/features/Train/features/Tabular/types/tabularTypes";
import { UseFormHandleSubmit, useForm } from "react-hook-form";
import React, { useEffect, useState } from "react";
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
import { useTrainMutation } from "../redux/tabularApi";
import { useRouter } from "next/router";
import { removeTrainspaceData } from "@/features/Train/redux/trainspaceSlice";

const TabularTrainspace = () => {
  const trainspace = useAppSelector(
    (state) => state.trainspace.current as TrainspaceData | undefined
  );
  const {
    handleSubmit,
    formState: { errors, isDirty },
    register,
  } = useForm<TrainspaceData>({
    defaultValues: trainspace,
  });
  const [step, setStep] = useState<number>(trainspace ? trainspace.step : 0);
  if (!trainspace) return <></>;
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
          activeStep={Math.min(
            trainspace.step,
            TRAINSPACE_SETTINGS.steps.length - 1
          )}
        >
          {TRAINSPACE_SETTINGS.steps.map((step, index) => (
            <Step key={step}>
              <StepButton onClick={() => setStep(index)}>
                {STEP_SETTINGS[step].name}
              </StepButton>
            </Step>
          ))}
        </Stepper>
      }
      trainspaceStep={
        <TrainspaceStepInner
          trainspace={trainspace}
          step={step}
          handleSubmit={handleSubmit}
          setStep={setStep}
          isModified={isDirty}
        />
      }
    />
  );
};

const TrainspaceStepInner = ({
  trainspace,
  step,
  handleSubmit,
  setStep,
  isModified,
}: {
  trainspace: TrainspaceData;
  step: number;
  handleSubmit: UseFormHandleSubmit<TrainspaceData>;
  setStep: React.Dispatch<React.SetStateAction<number>>;
  isModified: boolean;
}) => {
  const Component = STEP_SETTINGS[TRAINSPACE_SETTINGS.steps[step]].component;
  const [isStepModified, setIsStepModified] = useState<boolean>(false);
  const [train] = useTrainMutation();
  const dispatch = useAppDispatch();
  const router = useRouter();
  useEffect(() => {
    if (trainspace.step < TRAINSPACE_SETTINGS.steps.length)
      setStep(trainspace.step);
    else {
      train(trainspace)
        .unwrap()
        .then(({ trainspaceId }) => {
          router.push({ pathname: `/train/${trainspaceId}` }).then(() => {
            dispatch(removeTrainspaceData());
          });
        });
    }
  }, [trainspace]);
  if (!Component) return <></>;
  return (
    <Component
      renderStepperButtons={(submitTrainspace) => (
        <Stack
          direction={"row"}
          justifyContent={step > 0 ? "space-between" : "end"}
        >
          {step > 0 ? (
            <Button variant="outlined" onClick={() => setStep(step - 1)}>
              Previous
            </Button>
          ) : null}
          <Button
            variant={
              step < trainspace.step && !isStepModified && !isModified
                ? "outlined"
                : "contained"
            }
            onClick={() => {
              if (step < trainspace.step && !isStepModified && !isModified) {
                setStep(step + 1);
                return;
              }
              handleSubmit(submitTrainspace)();
            }}
          >
            {step < TRAINSPACE_SETTINGS.steps.length - 1 ? "Next" : "Train"}
          </Button>
        </Stack>
      )}
      setIsModified={setIsStepModified}
    />
  );
};

export default TabularTrainspace;
