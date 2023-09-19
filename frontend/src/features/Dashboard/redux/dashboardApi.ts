import { backendApi } from "@/common/redux/backendApi";
import { TrainResultsData } from "@/features/Train/types/trainTypes";

const dashboardApi = backendApi.injectEndpoints({
  endpoints: (builder) => ({
    getExecutionsData: builder.query<TrainResultsData[], void>({
      query: () => ({
        url: "/api/training/test",
        method: "GET",
        /*body: {
          user: auth.currentUser,
        },*/
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
  overrideExisting: true,
});

export const { useGetExecutionsDataQuery } = dashboardApi;
