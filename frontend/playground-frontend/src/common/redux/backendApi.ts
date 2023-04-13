import {
  BaseQueryApi,
  BaseQueryFn,
  FetchArgs,
  FetchBaseQueryError,
  createApi,
  fetchBaseQuery,
} from "@reduxjs/toolkit/query/react";
import { auth } from "../utils/firebase";
import { he } from "date-fns/locale";
import { FirebaseError } from "firebase/app";
import axios, { AxiosRequestConfig } from "axios";
import { MaybePromise } from "@reduxjs/toolkit/dist/query/tsHelpers";
import { TrainSpaceData } from "@/features/Dashboard/types/train_types";

const customFetchBaseQuery =
  ({
    prepareHeaders,
  }: {
    prepareHeaders?: (
      headers: Headers
    ) => MaybePromise<void | Headers> | undefined;
  }): BaseQueryFn<FetchArgs, unknown, FetchBaseQueryError> =>
  async (args, api, extraOptions) => {
    args.headers = new Headers();
    if (prepareHeaders) {
      try {
        await prepareHeaders(args.headers);
      } catch (e) {
        let err = e as FirebaseError;
        return {
          error: {
            status: "CUSTOM_ERROR",
            data: err,
            error: err.message,
          },
        };
      }
    }
    const baseQuery = fetchBaseQuery();
    return baseQuery(args, api, extraOptions);
  };

export const backendApi = createApi({
  reducerPath: "backendApi",
  baseQuery: customFetchBaseQuery({
    prepareHeaders: async (headers) => {
      if (auth.currentUser) {
        const token = await auth.currentUser.getIdToken();
        headers.set("Authorization", `Bearer ${token}`);
        headers.entries().next();
        return headers;
      }
    },
  }),
  endpoints: (builder) => ({
    getExecutionsData: builder.query<TrainSpaceData[], void>({
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
