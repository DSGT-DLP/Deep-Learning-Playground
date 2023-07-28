import {
  DATA_SOURCE,
  FileUploadData,
  ImageUploadData,
} from "@/features/Train/types/trainTypes";
import { Button, Radio, Typography } from "@mui/material";
import { Controller, UseFormReturn } from "react-hook-form";
import { formatDate } from "@/common/utils/dateFormat";
import { useGetDatasetFilesDataQuery } from "@/features/Train/redux/trainspaceApi";
import { DataGrid } from "@mui/x-data-grid";
import prettyBytes from "pretty-bytes";
import React from "react";
import TuiImageEditor from "./TuiImageEditor";

export const UploadImagePanel = ({
  dataSource,
  methods,
}: {
  dataSource: DATA_SOURCE;
  methods: UseFormReturn<ImageUploadData, unknown>;
}) => {
  const { data, refetch } = useGetDatasetFilesDataQuery({ dataSource });
  return (
    <>
      <TuiImageEditor dataSource={dataSource} />
      {methods.formState.errors.name && (
        <Typography color={"red"}>Please select a file</Typography>
      )}
      <Button variant="outlined" onClick={() => refetch()}>
        Refresh
      </Button>
      {data && <UploadDataGrid data={data} methods={methods} />}
    </>
  );
};

export const UploadDataGrid = ({
  data,
  methods,
}: {
  data: FileUploadData[];
  methods: UseFormReturn<ImageUploadData, unknown>;
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
