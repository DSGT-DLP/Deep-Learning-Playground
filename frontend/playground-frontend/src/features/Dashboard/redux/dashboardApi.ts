import { backendApi } from "@/common/redux/backendApi";
import { TrainResultsData } from "@/features/Train/types/trainTypes";
import { auth } from "@/common/utils/firebase";
import camelCase from "lodash.camelcase";

const dashboardApi = backendApi.injectEndpoints({
  endpoints: (builder) => ({
    getExecutionsData: builder.query<TrainResultsData[], void>({
      query: () => ({
        url: "/api/getTrainspaceData",
        method: "POST",
        body: {
          user: auth.currentUser,
        },
      }),
      transformResponse: (response: { record: string }) => {
        /*const data = JSON.parse(response.record);
        Object.entries(data).forEach(([key, value]) => {
          if (key === "created") {
            return [key, new Date(value as string)];
          }
          return [camelCase(key), value];
        });*/
        return [];
      },
    }),
  }),
});

export const { useGetExecutionsDataQuery } = dashboardApi;
