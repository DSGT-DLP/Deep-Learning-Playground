import { backendApi } from "@/common/redux/backendApi";
import { auth } from "@/common/utils/firebase";
import { FileUploadData } from "@/features/Train/types/trainTypes";
import { fetchBaseQuery } from "@reduxjs/toolkit/dist/query";

const trainspaceApi = backendApi
  .enhanceEndpoints({ addTagTypes: ["UserDatasetFilesData"] })
  .injectEndpoints({
    endpoints: (builder) => ({
      getDatasetFilesData: builder.query<FileUploadData[], void>({
        query: () => ({
          url: "/api/getUserDatasetFilesData",
          method: "POST",
          body: {
            user: auth.currentUser,
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
      uploadDatasetFile: builder.mutation<null, { file: File }>({
        queryFn: async ({ file }, _, __, baseQuery) => {
          const postObjResponse = await baseQuery({
            url: "/api/getUserDatasetFileUploadPresignedPostObj",
            method: "POST",
            body: {
              name: file.name,
              user: auth.currentUser,
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
    }),
    overrideExisting: true,
  });

export const { useGetDatasetFilesDataQuery, useUploadDatasetFileMutation } =
  trainspaceApi;
