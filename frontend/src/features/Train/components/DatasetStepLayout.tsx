import React from "react";
import { Controller, UseFormReturn } from "react-hook-form";
import {
  FileUploadData,
  DATA_SOURCE,
  DatasetData,
} from "@/features/Train/types/trainTypes";
import {
  Button,
  FormControl,
  FormControlLabel,
  FormLabel,
  Radio,
  RadioGroup,
  Stack,
  Tab,
  Tabs,
  Typography,
} from "@mui/material";
import { useAppSelector } from "@/common/redux/hooks";
import { ALL_STEP_SETTINGS } from "@/features/Train/constants/trainConstants";
import { DataGrid } from "@mui/x-data-grid";
import {
  useGetDatasetFilesDataQuery,
  useUploadDatasetFileMutation,
} from "@/features/Train/redux/trainspaceApi";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import { formatDate } from "@/common/utils/dateFormat";
import prettyBytes from "pretty-bytes";

export const DatasetStepTabLayout = ({
  currTab,
  setCurrTab,
  tabs,
  tabComponents,
  stepperButtons,
}: {
  currTab: string;
  setCurrTab: React.Dispatch<React.SetStateAction<string>>;
  tabs: { tabLabel: string; tabValue: string }[];
  tabComponents: { [tabValue: string]: React.ReactNode };
  stepperButtons: React.ReactNode;
}) => {
  return (
    <Stack spacing={3}>
      <Tabs
        value={currTab}
        onChange={(_, newValue: string) => {
          setCurrTab(newValue);
        }}
        aria-label="basic tabs example"
      >
        {tabs.map((tab) => (
          <Tab key={tab.tabValue} label={tab.tabLabel} value={tab.tabValue} />
        ))}
      </Tabs>
      {tabComponents[currTab]}
      {stepperButtons}
    </Stack>
  );
};

export const UploadDatasetPanel = ({
  dataSource,
  methods,
  acceptedTypes,
}: {
  dataSource: DATA_SOURCE;
  methods: UseFormReturn<DatasetData, unknown>;
  acceptedTypes: string;
}) => {
  const [uploadFile] = useUploadDatasetFileMutation();
  const { data, refetch } = useGetDatasetFilesDataQuery({ dataSource });
  return (
    <>
      {methods.formState.errors.name && (
        <Typography>Please select a file</Typography>
      )}
      <Stack direction={"row"} spacing={2}>
        <Button
          variant="contained"
          component="label"
          startIcon={<CloudUploadIcon />}
        >
          Upload
          <input
            type="file"
            accept={acceptedTypes}
            hidden
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                uploadFile({ dataSource: dataSource, file: e.target.files[0] });
              }
              e.target.value = "";
            }}
          />
        </Button>
        <Button variant="outlined" onClick={() => refetch()}>
          Refresh
        </Button>
      </Stack>
      {data && <UploadDataGrid data={data} methods={methods} />}
    </>
  );
};

export const UploadDataGrid = ({
  data,
  methods,
}: {
  data: FileUploadData[];
  methods: UseFormReturn<DatasetData, unknown>;
}) => {
  return (
    <Controller
      name="name"
      control={methods.control}
      rules={{ required: true }}
      render={({ field: { onChange, value } }) => (
        <DataGrid
          initialState={{
            sorting: {
              sortModel: [{ field: "lastModified", sort: "desc" }],
            },
          }}
          disableRowSelectionOnClick
          rows={data}
          hideFooter={true}
          getRowId={(row) => row.name}
          autoHeight
          disableColumnMenu
          sx={{ border: 0 }}
          columns={[
            {
              field: "radio",
              width: 75,
              filterable: false,
              sortable: false,
              hideable: false,
              disableColumnMenu: true,
              renderHeader: () => {
                return <></>;
              },
              renderCell: (params) => {
                return (
                  <Radio
                    value={params.row.name}
                    checked={value === params.row.name}
                    onChange={(e) => {
                      onChange(e);
                      methods.setValue("isDefaultDataset", false);
                    }}
                  />
                );
              },
            },
            { field: "name", headerName: "Name", flex: 3 },
            {
              field: "sizeInBytes",
              headerName: "Size",
              flex: 1,
              valueFormatter: (params) => prettyBytes(params.value),
            },
            {
              field: "lastModified",
              headerName: "Last Modified",
              flex: 2,
              valueFormatter: (params) => formatDate(new Date(params.value)),
            },
          ]}
        />
      )}
    />
  );
};

export const DefaultDatasetPanel = ({
  methods,
}: {
  methods: UseFormReturn<DatasetData, unknown>;
}) => {
  const trainspace = useAppSelector((state) => state.trainspace.current);
  if (!trainspace) return <></>;
  return (
    <FormControl>
      <FormLabel>Choose a Default Dataset</FormLabel>
      {methods.formState.errors.name && (
        <Typography>Please select a default dataset</Typography>
      )}
      <Controller
        name="name"
        control={methods.control}
        rules={{ required: true }}
        render={({ field: { onChange, value } }) => (
          <RadioGroup
            onChange={(e) => {
              onChange(e);
              methods.setValue("isDefaultDataset", true);
            }}
            value={value ?? ""}
          >
            {ALL_STEP_SETTINGS[trainspace.dataSource]["DATASET"][
              "defaultDatasets"
            ].map((defaultDataset) => (
              <FormControlLabel
                key={defaultDataset.value}
                value={defaultDataset.value}
                control={<Radio />}
                label={defaultDataset.label}
              />
            ))}
          </RadioGroup>
        )}
      />
    </FormControl>
  );
};
