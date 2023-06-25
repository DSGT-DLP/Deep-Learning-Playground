import { backendApi } from "@/common/redux/backendApi";
import { TrainspaceData } from "../types/detectionTypes";
import { auth } from "@/common/utils/firebase";

const detectionApi = backendApi.injectEndpoints({
  endpoints: (builder) => ({
    detection: builder.mutation<{ trainspaceId: string }, TrainspaceData<"TRAIN">>({
      query: (trainspaceData) => ({
        url: "/api/train/object-detection",
        method: "POST",
        body: {
          user: auth.currentUser,
          name: trainspaceData.name,
          image_data: {
            name: trainspaceData.imageData.name,
          },
          parameters_data: {
            problem_type: trainspaceData.parameterData.detectionType,
            detection_problem_type: trainspaceData.parameterData.detectionProblemType,
            transforms: trainspaceData.parameterData.transforms,
          },
          review_data: {
            notification_email:
              trainspaceData.reviewData.notificationEmail ?? null,
            notification_phone_number:
              trainspaceData.reviewData.notificationPhoneNumber ?? null,
          },
        },
      }),
      transformResponse: (response: { trainspace_id: string }) => {
        return {
          trainspaceId: response.trainspace_id,
        };
      },
    }),
  }),
  overrideExisting: true,
});

export const { useDetectionMutation } = detectionApi;
