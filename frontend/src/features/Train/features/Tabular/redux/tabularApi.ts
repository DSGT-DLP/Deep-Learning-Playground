import { backendApi } from "@/common/redux/backendApi";
import { TrainspaceData } from "../types/tabularTypes";

const tabularApi = backendApi.injectEndpoints({
  endpoints: (builder) => ({
    trainTabular: builder.mutation<
      { trainspaceId: string },
      TrainspaceData<"TRAIN">
    >({
      query: (trainspaceData) => ({
        url: "/api/training/tabular",
        method: "POST",
        body: {
          name: trainspaceData.name,
          data_source: trainspaceData.dataSource,
          target: trainspaceData.parameterData.targetCol,
          features: trainspaceData.parameterData.features,
          default: trainspaceData.datasetData.isDefaultDataset
            ? trainspaceData.datasetData.name
            : undefined,
          problem_type: trainspaceData.parameterData.problemType,
          criterion: trainspaceData.parameterData.criterion,
          optimizer_name: trainspaceData.parameterData.optimizerName,
          shuffle: trainspaceData.parameterData.shuffle,
          epochs: trainspaceData.parameterData.epochs,
          test_size: trainspaceData.parameterData.testSize,
          batch_size: trainspaceData.parameterData.batchSize,
          user_arch: trainspaceData.parameterData.layers,
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

export const { useTrainTabularMutation } = tabularApi;
