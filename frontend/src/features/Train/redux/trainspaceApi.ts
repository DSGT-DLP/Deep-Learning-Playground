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
          url: `/api/lambda/datasets/user/${dataSource}`,
        }),
        providesTags: ["UserDatasetFilesData"],
        transformResponse: (response: {
          data: {
            name: string;
            type: string;
            last_modified: string;
            size: string;
          }[];
        }) => {
          return response.data.map<FileUploadData>((fileObj) => ({
            name: fileObj.name,
            contentType: fileObj.type,
            lastModified: fileObj.last_modified,
            sizeInBytes: parseInt(fileObj.size),
          }));
        },
      }),
      uploadDatasetFile: builder.mutation<
        null,
        { dataSource: DATA_SOURCE; file: File; url: string }
      >({
        queryFn: async ({ dataSource, file }, _, __, baseQuery) => {
          const getObjResponse = await baseQuery({
            url: `/api/lambda/datasets/user/${dataSource}/${file.name}/presigned_upload_url`,
            method: "GET",
          });
          if (getObjResponse.error) {
            return { error: getObjResponse.error };
          }
          const getObj = getObjResponse.data as {
            data: string;
            message: string;
          };
          const getObjUrl = getObj.data;
          const uploadQuery = fetchBaseQuery();
          const response = await uploadQuery(
            {
              url: getObjUrl,
              method: "PUT",
              body: file,
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
            ? `/api/training/datasets/default/${dataset.name}/columns`
            : `/api/lambda/datasets/user/${dataSource}/${dataset.name}/columns`,
          method: "GET",
          /*body: {
            data_source: dataSource.toLowerCase(),
            name: dataset.isDefaultDataset ? undefined : dataset.name,
            using_default_dataset: dataset.isDefaultDataset
              ? dataset.name
              : undefined,
          },*/
        }),
        transformResponse: (response: { data: string[] }) => {
          return response.data;
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
