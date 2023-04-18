import React from "react";
import { Controller, UseFormReturn, useForm } from "react-hook-form";
import { BaseTrainspaceData, DefaultUploadData } from "../types/trainTypes";
import {
  FormControl,
  FormControlLabel,
  FormLabel,
  Radio,
  RadioGroup,
  Stack,
  Tab,
  Tabs,
} from "@mui/material";
import { useAppSelector } from "@/common/redux/hooks";
import { DATA_SOURCE_SETTINGS } from "../constants/trainConstants";

const UploadStep = ({
  renderStepperButtons,
}: {
  renderStepperButtons: (handleStepSubmit: () => void) => React.ReactNode;
}) => {
  const [currTab, setCurrTab] = React.useState("upload-dataset");
  const defaultDatasetMethods = useForm<DefaultUploadData>();
  return (
    <Stack spacing={3}>
      <Tabs
        value={currTab}
        onChange={(_, newValue: string) => {
          setCurrTab(newValue);
        }}
        aria-label="basic tabs example"
      >
        <Tab label="Recently Uploaded Datasets" value="upload-dataset" />
        <Tab label="Default Datasets" value="default-dataset" />
      </Tabs>
      {currTab === "upload-dataset" ? (
        <UploadDatasetPanel />
      ) : (
        <DefaultDatasetPanel methods={defaultDatasetMethods} />
      )}
      {renderStepperButtons(() => {
        console.log("hi");
      })}
    </Stack>
  );
};

const UploadDatasetPanel = () => {
  return <></>;
};

const DefaultDatasetPanel = ({
  methods,
}: {
  methods: UseFormReturn<DefaultUploadData, unknown>;
}) => {
  const trainspace = useAppSelector((state) => state.trainspace.current);
  if (!trainspace) return <></>;
  return (
    <FormControl>
      <FormLabel>Choose a Default Dataset</FormLabel>
      <Controller
        name="dataSetName"
        control={methods.control}
        rules={{ required: true }}
        render={({ field: { onChange, value } }) => (
          <RadioGroup onChange={onChange} value={value}>
            {DATA_SOURCE_SETTINGS[trainspace.dataSource].defaultDatasets.map(
              (defaultDataset) => (
                <FormControlLabel
                  key={defaultDataset.value}
                  value={defaultDataset.value}
                  control={<Radio />}
                  label={defaultDataset.label}
                />
              )
            )}
          </RadioGroup>
        )}
      />
    </FormControl>
  );
};

export default UploadStep;
