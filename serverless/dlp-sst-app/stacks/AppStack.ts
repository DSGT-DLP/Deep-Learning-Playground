import { Api, StackContext } from "sst/constructs";

export function AppStack({ stack }: StackContext) {
  const api = new Api(stack, "Api", {
    routes: {
      "GET /lambda/get_signed_upload_url":
        "packages/functions/src/get_signed_upload_url.handler",
    },
  });
  // Show the API endpoint in the output
  stack.addOutputs({
    ApiEndpoint: api.url,
  });
}
