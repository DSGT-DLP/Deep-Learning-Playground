import { backendApi } from "@/common/redux/backendApi";
import { TrainspaceData } from "../types/detectionTypes";
import { auth } from "@/common/utils/firebase";

const tabularApi = backendApi.injectEndpoints({
  endpoints: (builder) => ({
    train: builder.mutation<{ trainspaceId: string }, TrainspaceData<"TRAIN">>({
      query: (trainspaceData) => ({
        url: "/api/train/tabular-run",
        method: "POST",
        body: {
          user: auth.currentUser,
          name: trainspaceData.name,
          dataset_data: {
            name: trainspaceData.datasetData.name,
            is_default_dataset: trainspaceData.datasetData.isDefaultDataset,
          },
          parameters_data: {
            detection_problem_type: trainspaceData.parameterData.problemType,

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

export const { useTrainMutation } = tabularApi;
