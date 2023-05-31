import { backendApi } from "@/common/redux/backendApi";
import { TrainspaceData } from "../types/tabularTypes";
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
            target_col: trainspaceData.parameterData.targetCol,
            features: trainspaceData.parameterData.features,
            problem_type: trainspaceData.parameterData.problemType,
            criterion: trainspaceData.parameterData.criterion,
            optimizer_name: trainspaceData.parameterData.optimizerName,
            shuffle: trainspaceData.parameterData.shuffle,
            epochs: trainspaceData.parameterData.epochs,
            test_size: trainspaceData.parameterData.testSize,
            batch_size: trainspaceData.parameterData.batchSize,
            layers: trainspaceData.parameterData.layers,
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
