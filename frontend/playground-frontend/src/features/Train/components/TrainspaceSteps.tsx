import { useAppSelector } from "@/common/redux/hooks";
import { ContentCopy } from "@mui/icons-material";
import {
  Card,
  Container,
  Grid,
  IconButton,
  Step,
  StepButton,
  Stepper,
} from "@mui/material";
import React from "react";
import SyntaxHighlighter from "react-syntax-highlighter";
import { a11yLight } from "react-syntax-highlighter/dist/cjs/styles/hljs";
import {
  DATA_SOURCE_SETTINGS,
  STEPS_SETTINGS,
} from "@/features/Train/constants/trainConstants";
import { ALL_STEPS } from "../types/trainTypes";
const TrainspaceSteps = () => {
  const trainspace = useAppSelector((state) => state.trainspace.current);
  if (!trainspace) return <></>;
  return (
    <Container style={{ marginTop: "75px", height: "100vh" }}>
      <Grid
        container
        spacing={20}
        direction="row"
        justifyContent="center"
        alignItems={"stretch"}
        sx={{ height: "100%" }}
      >
        <Grid item xs={5}>
          <Card style={{ height: "75%", position: "relative" }}>
            <SyntaxHighlighter
              customStyle={{ height: "100%", fontSize: "1rem", paddingTop: 40 }}
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
        <Grid item xs={7}>
          <Stepper activeStep={0}>
            {DATA_SOURCE_SETTINGS[trainspace.dataSource].steps.map((step) => (
              <Step key={step}>
                <StepButton>
                  {STEPS_SETTINGS[step as ALL_STEPS].name}
                </StepButton>
              </Step>
            ))}
          </Stepper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default TrainspaceSteps;
