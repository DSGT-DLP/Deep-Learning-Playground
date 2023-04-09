import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";
import { auth } from "../utils/firebase";
import { Execution } from "@/pages/dashboard";
export const backendApi = createApi({
  reducerPath: "backendApi",
  baseQuery: fetchBaseQuery({
    baseUrl: "http://127.0.0.1:8000",
    prepareHeaders: async (headers, { extra }) => {
      if (auth.currentUser) {
        const token = await auth.currentUser.getIdToken(true);
        headers.set("authorization", `Bearer ${token}`);
        return headers;
      }
    },
  }),
  endpoints: (builder) => ({
    getExecutionsData: builder.query<Execution[], void>({
      query: () => "/api/getExecutionsFilesPresignedUrls",
      transformResponse: (response: { record: string }) => {
        return JSON.parse(response.record);
      },
    }),
    getExecutionsFilesPresignedUrls: builder.query<unknown, void>({
      query: () => "/api/getExecutionsFilesPresignedUrls",
    }),
  }),
});

export const { useGetExecutionsDataQuery } = backendApi;
