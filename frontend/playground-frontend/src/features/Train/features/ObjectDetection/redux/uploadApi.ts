import { backendApi } from "@/common/redux/backendApi";
import { auth } from "@/common/utils/firebase";

const feedbackApi = backendApi.injectEndpoints({
  endpoints: (builder) => ({
    sendUploadData: builder.query({
      query: (args) => {
        const { file } = args;
        return {
        url: "/api/s3/upload",
        method: "POST",
        body: {
          file: file
        },
      }
      },
    }),
  }),
  overrideExisting: true
});

export const { useLazySendUploadDataQuery } = feedbackApi;
