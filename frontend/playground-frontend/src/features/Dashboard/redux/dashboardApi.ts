import { backendApi } from "@/common/redux/backendApi";
import { TrainSpaceData } from "../types/trainTypes";
import { auth } from "@/common/utils/firebase";

const dashboardApi = backendApi.injectEndpoints({
  endpoints: (builder) => ({
    getExecutionsData: builder.query<TrainSpaceData[], void>({
      query: () => ({
        url: "/api/getExecutionsData",
        method: "POST",
        body: {
          user: auth.currentUser,
        },
      }),
      transformResponse: (response: { record: string }) => {
        return JSON.parse(response.record);
      },
    }),
  }),
});

export const { useGetExecutionsDataQuery } = dashboardApi;
