import { ContentCopy } from "@mui/icons-material";
import { Card, Container, Grid, IconButton, Stack } from "@mui/material";
import React from "react";
import SyntaxHighlighter from "react-syntax-highlighter";
import { a11yLight } from "react-syntax-highlighter/dist/cjs/styles/hljs";
import { BaseTrainspaceData } from "../types/trainTypes";
const TrainspaceLayout = ({
  code,
  nameField,
  stepper,
  trainspaceStep,
}: {
  code: string;
  nameField: React.ReactNode;
  stepper: React.ReactNode;
  trainspaceStep: React.ReactNode;
}) => {
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
            >
              {code}
            </SyntaxHighlighter>
            <IconButton style={{ position: "absolute", right: 15, top: 5 }}>
              <ContentCopy />
            </IconButton>
          </Card>
        </Grid>
        <Grid item sm={7}>
          <Stack direction="column" spacing={5}>
            {nameField}
            {stepper}
            {trainspaceStep}
          </Stack>
        </Grid>
      </Grid>
    </Container>
  );
};

export const TrainspaceStep = ({
  stepComponent,
  renderStepperButtons,
}: {
  stepComponent: React.FC<{
    renderStepperButtons: (
      handleStepSubmit: (data: BaseTrainspaceData) => void
    ) => React.ReactNode;
  }>;
  renderStepperButtons: (
    submitTrainspace: (data: BaseTrainspaceData) => void
  ) => React.ReactNode;
}) => {
  const StepComponent = stepComponent;
  return <StepComponent renderStepperButtons={renderStepperButtons} />;
};

export default TrainspaceLayout;
