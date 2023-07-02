import {
  DATA_SOURCE,
  FileUploadData,
  ImageUploadData,
} from "@/features/Train/types/trainTypes";
import { Button, Radio, Stack, Typography } from "@mui/material";
import { Controller, UseFormReturn } from "react-hook-form";

import { formatDate } from "@/common/utils/dateFormat";
import {
  useGetDatasetFilesDataQuery,
  useUploadDatasetFileMutation,
} from "@/features/Train/redux/trainspaceApi";
import { DataGrid } from "@mui/x-data-grid";
import prettyBytes from "pretty-bytes";
import FilerobotImageEditor, {
  TABS,
  TOOLS,
} from "react-filerobot-image-editor";
import React from "react";

const dataURLtoFile = (dataurl: string, filename: string) => {
  const arr = dataurl.split(",");
  if (arr.length === 0) {
    return new File([""], filename);
  }
  const matched = arr[0].match(/:(.*?);/);
  const mime = matched ? matched[1] : undefined;
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n) {
    u8arr[n - 1] = bstr.charCodeAt(n - 1);
    n -= 1; // to make eslint happy
  }
  return new File([u8arr], filename, { type: mime });
};

export const UploadImagePanel = ({
  dataSource,
  methods,
}: {
  dataSource: DATA_SOURCE;
  methods: UseFormReturn<ImageUploadData, unknown>;
}) => {
  const [uploadFile] = useUploadDatasetFileMutation();
  const { data, refetch } = useGetDatasetFilesDataQuery({ dataSource });
  return (
    <>
      {methods.formState.errors.name && (
        <Typography>Please select a file</Typography>
      )}
      <Stack direction={"row"} spacing={2}>
        <FilerobotImageEditor
          source="https://scaleflex.airstore.io/demo/stephen-walker-unsplash.jpg"
          onSave={(editedImageObject: unknown) => {
            const file = dataURLtoFile(
              editedImageObject.imageBase64,
              editedImageObject.fullName
            );
            uploadFile({ dataSource, file });
          }}
          annotationsCommon={{
            fill: "#ff0000",
          }}
          Text={{ text: "Filerobot..." }}
          Rotate={{ angle: 90, componentType: "slider" }}
          Crop={{
            presetsItems: [
              {
                titleKey: "classicTv",
                descriptionKey: "4:3",
                ratio: (4 / 3).toString(),
              },
              {
                titleKey: "cinemascope",
                descriptionKey: "21:9",
                ratio: (21 / 9).toString(),
              },
            ],
            presetsFolders: [
              {
                titleKey: "socialMedia",

                // icon: Social, // optional, Social is a React Function component. Possible (React Function component, string or HTML Element)
                groups: [
                  {
                    titleKey: "facebook",
                    items: [
                      {
                        titleKey: "profile",
                        width: 180,
                        height: 180,
                        descriptionKey: "fbProfileSize",
                      },
                      {
                        titleKey: "coverPhoto",
                        width: 820,
                        height: 312,
                        descriptionKey: "fbCoverPhotoSize",
                      },
                    ],
                  },
                ],
              },
            ],
          }}
          tabsIds={[TABS.ADJUST, TABS.ANNOTATE, TABS.WATERMARK]} // or {['Adjust', 'Annotate', 'Watermark']}
          defaultTabId={TABS.ANNOTATE} // or 'Annotate'
          defaultToolId={TOOLS.TEXT} // or 'Text'
          savingPixelRatio={0}
          previewPixelRatio={0}
        />
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
