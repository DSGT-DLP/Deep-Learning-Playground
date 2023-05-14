import { backendApi } from "@/common/redux/backendApi";
import { auth } from "@/common/utils/firebase";

const feedbackApi = backendApi.injectEndpoints({
  endpoints: (builder) => ({
    sendFeedbackData: builder.query({
      query: (args) => {
        const { email_address, subject, body_text } = args;
        return {
        url: "/aws/sendEmail",
        method: "POST",
        body: {
          user: auth.currentUser,
          email_address: email_address,
          subject: subject,
          body_text: body_text,
        },
      }
      },
    }),
  }),
  overrideExisting: true
});

export const { useLazySendFeedbackDataQuery } = feedbackApi;
