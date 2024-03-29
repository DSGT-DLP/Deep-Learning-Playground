import { backendApi } from "@/common/redux/backendApi";
import { TrainspaceData } from "../types/imageTypes";

const imageApi = backendApi.injectEndpoints({
  endpoints: (builder) => ({
    trainImage: builder.mutation<
      { trainspaceId: string },
      TrainspaceData<"TRAIN">
    >({
      query: (trainspaceData) => ({
        url: "/api/train/img-run",
        method: "POST",
        body: {
          name: trainspaceData.name,
          data_source: trainspaceData.dataSource,
          dataset_data: {
            name: trainspaceData.datasetData.name,
            is_default_dataset: trainspaceData.datasetData.isDefaultDataset,
          },
          parameters_data: {
            criterion: trainspaceData.parameterData.criterion,
            optimizer_name: trainspaceData.parameterData.optimizerName,
            shuffle: trainspaceData.parameterData.shuffle,
            epochs: trainspaceData.parameterData.epochs,
            batch_size: trainspaceData.parameterData.batchSize,
            train_transforms: trainspaceData.parameterData.trainTransforms,
            test_transforms: trainspaceData.parameterData.testTransforms,
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

export const { useTrainImageMutation } = imageApi;
