import { ContentCopy } from "@mui/icons-material";
import { Card, Container, Grid, IconButton, Stack } from "@mui/material";
import React from "react";
import SyntaxHighlighter from "react-syntax-highlighter";
import { a11yLight } from "react-syntax-highlighter/dist/cjs/styles/hljs";
import { DATA_SOURCE } from "../types/trainTypes";

const TrainspaceLayout = ({
  code,
  dataSource,
  nameField,
  stepper,
  trainspaceStep,
}: {
  code: string;
  dataSource: DATA_SOURCE;
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
        {dataSource !== "OBJECT_DETECTION" ? (
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
        ) : (
          <></>
        )}
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

export default TrainspaceLayout;
