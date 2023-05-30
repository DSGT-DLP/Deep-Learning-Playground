import { backendApi } from "@/common/redux/backendApi";
import { auth } from "@/common/utils/firebase";

const feedbackApi = backendApi.injectEndpoints({
  endpoints: (builder) => ({
    sendObjectDetectionData: builder.query({
      query: (args) => {
        const { problem_type, detection_type, transforms } = args;
        return {
        url: "/api/train/object-detection",
        method: "POST",
        body: {
          user: auth.currentUser,
          problem_type: problem_type,
          detection_type: detection_type,
          transforms: transforms,
        },
      }
      },
    }),
  }),
  overrideExisting: true
});

export const { useLazySendObjectDetectionDataQuery } = feedbackApi;
