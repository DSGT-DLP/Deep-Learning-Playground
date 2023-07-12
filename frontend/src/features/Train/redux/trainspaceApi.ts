import { backendApi } from "@/common/redux/backendApi";
import {
  DATA_SOURCE,
  DatasetData,
  FileUploadData,
} from "@/features/Train/types/trainTypes";
import { fetchBaseQuery } from "@reduxjs/toolkit/dist/query";

const trainspaceApi = backendApi
  .enhanceEndpoints({ addTagTypes: ["UserDatasetFilesData"] })
  .injectEndpoints({
    endpoints: (builder) => ({
      getDatasetFilesData: builder.query<
        FileUploadData[],
        { dataSource: DATA_SOURCE }
      >({
        query: ({ dataSource }) => ({
          url: "/api/s3/getUserDatasetFilesData",
          method: "POST",
          body: {
            data_source: dataSource.toLowerCase(),
          },
        }),
        providesTags: ["UserDatasetFilesData"],
        transformResponse: (response: { data: string }) => {
          const data: {
            name: string;
            type: string;
            last_modified: string;
            size: string;
          }[] = JSON.parse(response.data);
          return data.map<FileUploadData>((fileObj) => ({
            name: fileObj.name,
            contentType: fileObj.type,
            lastModified: fileObj.last_modified,
            sizeInBytes: parseInt(fileObj.size),
          }));
        },
      }),
      uploadDatasetFile: builder.mutation<
        null,
        { dataSource: DATA_SOURCE; file: File }
      >({
        queryFn: async ({ dataSource, file }, _, __, baseQuery) => {
          const postObjResponse = await baseQuery({
            url: "/api/s3/getUserDatasetFileUploadPresignedPostObj",
            method: "POST",
            body: {
              name: file.name,
              data_source: dataSource.toLowerCase(),
            },
          });
          if (postObjResponse.error) {
            return { error: postObjResponse.error };
          }
          const postObj = postObjResponse.data as {
            presigned_post_obj: { url: string; fields: Record<string, string> };
          };
          const formData = new FormData();
          for (const [key, value] of Object.entries(
            postObj.presigned_post_obj.fields
          )) {
            formData.append(key, value);
          }
          formData.append("file", file);
          const uploadQuery = fetchBaseQuery();
          const response = await uploadQuery(
            {
              url: postObj.presigned_post_obj.url,
              method: "POST",
              body: formData,
            },
            _,
            __
          );
          if (response.error) {
            return { error: response.error };
          }
          return { data: null };
        },
        invalidatesTags: ["UserDatasetFilesData"],
      }),
      getColumnsFromDataset: builder.query<
        string[],
        { dataSource: DATA_SOURCE; dataset: DatasetData }
      >({
        query: ({ dataSource, dataset }) => ({
          url: dataset.isDefaultDataset
            ? "/api/dataset/defaultDataset"
            : "/api/dataset/getColumnsFromDatasetFile",
          method: "POST",
          body: {
            data_source: dataSource.toLowerCase(),
            name: dataset.isDefaultDataset ? undefined : dataset.name,
            using_default_dataset: dataset.isDefaultDataset
              ? dataset.name
              : undefined,
          },
        }),
        transformResponse: (response: { columns: string }) => {
          return JSON.parse(response.columns);
        },
      }),
    }),
    overrideExisting: true,
  });

export const {
  useGetDatasetFilesDataQuery,
  useUploadDatasetFileMutation,
  useLazyGetColumnsFromDatasetQuery,
} = trainspaceApi;
