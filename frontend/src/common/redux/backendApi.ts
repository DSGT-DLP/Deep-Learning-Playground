import {
  BaseQueryFn,
  FetchArgs,
  FetchBaseQueryError,
  createApi,
  fetchBaseQuery,
} from "@reduxjs/toolkit/query/react";
import { auth } from "../utils/firebase";
import { FirebaseError } from "firebase/app";
import { MaybePromise } from "@reduxjs/toolkit/dist/query/tsHelpers";

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
        const err = e as FirebaseError;
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
        console.log(token);
        headers.set("Authorization", `Bearer ${token}`);
        headers.entries().next();
        return headers;
      }
    },
  }),
  endpoints: (builder) => ({
    getBeginnerMessage: builder.query<{ result: string }, void>({
      query: () => {
        return {
          url: "/api/training/beginner_endpoint",
        };
      },
    }),
  }),
});
