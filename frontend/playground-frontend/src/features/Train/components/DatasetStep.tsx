import React from "react";
import {
  Controller,
  ControllerFieldState,
  ControllerRenderProps,
  FieldValues,
  UseFormReturn,
  UseFormStateReturn,
  useForm,
} from "react-hook-form";
import {
  BaseTrainspaceData,
  DefaultDatasetData,
  FileUploadData,
  FileDatasetData,
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
import { useAppDispatch, useAppSelector } from "@/common/redux/hooks";
import {
  DATA_SOURCE_SETTINGS,
  setTrainspaceDataset,
} from "../constants/trainConstants";
import { DataGrid } from "@mui/x-data-grid";
import {
  useGetDatasetFilesDataQuery,
  useUploadDatasetFileMutation,
} from "@/features/Train/redux/trainspaceApi";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import { formatDate } from "@/common/utils/dateFormat";
import prettyBytes from "pretty-bytes";
import { setTrainspaceData } from "../redux/trainspaceSlice";

const DatasetStep = ({
  renderStepperButtons,
}: {
  renderStepperButtons: (
    submitTrainspace: (data: BaseTrainspaceData) => void
  ) => React.ReactNode;
}) => {
  const [currTab, setCurrTab] = React.useState("upload-dataset");
  const defaultDatasetMethods = useForm<DefaultDatasetData>();
  const uploadDatasetMethods = useForm<FileDatasetData>();
  const dispatch = useAppDispatch();
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
        <UploadDatasetPanel methods={uploadDatasetMethods} />
      ) : (
        <DefaultDatasetPanel methods={defaultDatasetMethods} />
      )}
      {renderStepperButtons((trainspaceData) => {
        if (currTab === "upload-dataset") {
          uploadDatasetMethods.handleSubmit((data) => {
            setTrainspaceDataset(trainspaceData, data);
          })();
        } else {
          defaultDatasetMethods.handleSubmit((data) => {
            setTrainspaceDataset(trainspaceData, data);
          })();
        }
        dispatch(setTrainspaceData(trainspaceData));
      })}
    </Stack>
  );
};

const UploadDatasetPanel = ({
  methods,
}: {
  methods: UseFormReturn<FileDatasetData, unknown>;
}) => {
  //const [getDatasetUploadPresignedUrl] =
  //  useGetDatasetUploadPresignedUrlMutation();
  const [uploadFile] = useUploadDatasetFileMutation();
  const { data, refetch } = useGetDatasetFilesDataQuery();
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
            hidden
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                console.log(e.target.files[0]);

                uploadFile({ file: e.target.files[0] });
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

const UploadDataGrid = ({
  data,
  methods,
}: {
  data: FileUploadData[];
  methods: UseFormReturn<FileDatasetData, unknown>;
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
                    onChange={onChange}
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

const DefaultDatasetPanel = ({
  methods,
}: {
  methods: UseFormReturn<DefaultDatasetData, unknown>;
}) => {
  const trainspace = useAppSelector((state) => state.trainspace.current);
  if (!trainspace) return <></>;
  return (
    <FormControl>
      <FormLabel>Choose a Default Dataset</FormLabel>
      {methods.formState.errors.dataSetName && (
        <Typography>Please select a default dataset</Typography>
      )}
      <Controller
        name="dataSetName"
        control={methods.control}
        rules={{ required: true }}
        render={({ field: { onChange, value } }) => (
          <RadioGroup onChange={onChange} value={value ?? ""}>
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

export default DatasetStep;
