import {
  BaseQueryFn,
  FetchArgs,
  FetchBaseQueryError,
  createApi,
  fetchBaseQuery,
} from "@reduxjs/toolkit/query/react";
import { auth } from "../utils/firebase";
import { Execution } from "@/pages/dashboard";
import { he } from "date-fns/locale";

export const backendApi = createApi({
  reducerPath: "backendApi",
  baseQuery: fetchBaseQuery({
    mode: "cors",
    prepareHeaders: async (headers) => {
      if (auth.currentUser) {
        const token = await auth.currentUser.getIdToken(true);
        console.log(token);
        headers.set("accepts", "application/json");
        headers.set("Authorization", `Bearer ${token}`);
        headers.entries().next();
        console.log(Array.from(headers.entries()));
        return headers;
      }
    },
  }),
  endpoints: (builder) => ({
    getExecutionsData: builder.query<Execution[], void>({
      query: () => ({
        url: "/api/getExecutionsData",
        method: "POST",
        body: {
          user: auth.currentUser,
        },
      }),
      transformResponse: (response: { record: string }) => {
        return JSON.parse(response.record);
      },
    }),
  }),
});

export const { useGetExecutionsDataQuery } = backendApi;
