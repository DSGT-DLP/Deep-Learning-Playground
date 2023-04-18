import { useAppSelector } from "@/common/redux/hooks";
import { ContentCopy } from "@mui/icons-material";
import {
  Button,
  Card,
  Container,
  Grid,
  IconButton,
  Stack,
  Step,
  StepButton,
  Stepper,
  TextField,
} from "@mui/material";
import React, { useEffect } from "react";
import SyntaxHighlighter from "react-syntax-highlighter";
import { a11yLight } from "react-syntax-highlighter/dist/cjs/styles/hljs";
import {
  DATA_SOURCE_SETTINGS,
  STEPS_SETTINGS,
} from "@/features/Train/constants/trainConstants";
import { ALL_STEPS, BaseTrainspaceData } from "../types/trainTypes";
import { useForm } from "react-hook-form";
const TrainspaceSteps = () => {
  const trainspace = useAppSelector((state) => state.trainspace.current);
  if (!trainspace) return <></>;
  const {
    handleSubmit,
    formState: { errors },
    register,
  } = useForm<BaseTrainspaceData>({
    defaultValues: trainspace,
  });
  return (
    <Container
      style={{ marginTop: "75px", marginBottom: "75px", minHeight: "100vh" }}
    >
      <Grid
        container
        spacing={20}
        direction="row"
        justifyContent="center"
        alignItems={"stretch"}
        sx={{ height: "100%" }}
      >
        <Grid item sm={5}>
          <Card style={{ height: "75vh", position: "sticky", top: "12.5vh" }}>
            <SyntaxHighlighter
              customStyle={{
                height: "100%",
                fontSize: "1rem",
                paddingTop: 40,
              }}
              language="python"
              showLineNumbers
              wrapLongLines
              style={a11yLight}
            >{`#Code to train your model locally\nprint("Hello World")`}</SyntaxHighlighter>
            <IconButton style={{ position: "absolute", right: 15, top: 5 }}>
              <ContentCopy />
            </IconButton>
          </Card>
        </Grid>
        <Grid item sm={7}>
          <Stack direction="column" spacing={5}>
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
            <Stepper
              activeStep={DATA_SOURCE_SETTINGS[
                trainspace.dataSource
              ].steps.indexOf(trainspace.step)}
            >
              {DATA_SOURCE_SETTINGS[trainspace.dataSource].steps.map((step) => (
                <Step key={step}>
                  <StepButton>
                    {STEPS_SETTINGS[step as ALL_STEPS].name}
                  </StepButton>
                </Step>
              ))}
            </Stepper>
            <TrainspaceStep
              step={trainspace.step as ALL_STEPS}
              renderStepperButtons={() => (
                <Stack direction={"row"} justifyContent={"space-between"}>
                  <Button>Previous</Button>
                  <Button>Next</Button>
                </Stack>
              )}
            />
          </Stack>
        </Grid>
      </Grid>
    </Container>
  );
};

const TrainspaceStep = ({
  step,
  renderStepperButtons,
}: {
  step: ALL_STEPS;
  renderStepperButtons: (handleStepSubmit: () => void) => React.ReactNode;
}) => {
  const StepComponent = STEPS_SETTINGS[step].stepComponent;
  return <StepComponent renderStepperButtons={renderStepperButtons} />;
};

export default TrainspaceSteps;
