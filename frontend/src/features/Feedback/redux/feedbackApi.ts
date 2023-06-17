import { backendApi } from "@/common/redux/backendApi";

const feedbackApi = backendApi.injectEndpoints({
  endpoints: (builder) => ({
    sendFeedbackData: builder.query({
      query: (args) => {
        const { email_address, subject, body_text } = args;
        return {
          url: "/api/aws/sendEmail",
          method: "POST",
          body: {
            email_address,
            subject,
            body_text,
          },
        };
      },
    }),
  }),
  overrideExisting: true,
});

export const { useLazySendFeedbackDataQuery } = feedbackApi;
