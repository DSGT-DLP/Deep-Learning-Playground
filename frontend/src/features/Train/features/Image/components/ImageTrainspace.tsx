import { useAppDispatch, useAppSelector } from "@/common/redux/hooks";
import { TrainspaceData } from "@/features/Train/features/Image/types/imageTypes";
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
  CircularProgress 
} from "@mui/material";
import {
  TRAINSPACE_SETTINGS,
  STEP_SETTINGS,
} from "../constants/imageConstants";
import { useTrainImageMutation } from "../redux/imageApi";
import { useRouter } from "next/router";
import { removeTrainspaceData } from "@/features/Train/redux/trainspaceSlice";

const ImageTrainspace = () => {
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
  const [train] = useTrainImageMutation();
  const[isButtonClicked, setIsButtonClicked] = useState<boolean>(false);
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
          {isButtonClicked ? 
            <Button disabled variant={
                step < trainspace.step && !isStepModified && !isModified
                  ? "outlined"
                  : "contained"
              }
            >
              {step < TRAINSPACE_SETTINGS.steps.length - 1 ? "Next" : "Train"}
            </Button> : 
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
              if (step === TRAINSPACE_SETTINGS.steps.length - 1) {
                setIsButtonClicked(true);
              }

              handleSubmit(submitTrainspace)();
            }}
          >
            {step < TRAINSPACE_SETTINGS.steps.length - 1 ? "Next" : "Train"}
          </Button>
      }
      {
        isButtonClicked ? <CircularProgress/> : null
      }
        </Stack>
      )}
      setIsModified={setIsStepModified}
    />
  );
};

export default ImageTrainspace;
