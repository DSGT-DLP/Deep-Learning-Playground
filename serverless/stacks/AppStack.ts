import { Api, Auth, StackContext } from "sst/constructs";

export function AppStack({ stack }: StackContext) {
  const api = new Api(stack, "Api", {
    /*authorizers: {
      FirebaseAuthorizer: {
        type: "jwt",
        jwt: {
          issuer: "",
          audience: [""],
        },
      },
    },
    defaults: {
      authorizer: "FirebaseAuthorizer",
    },*/
    routes: {
      "GET /lambda/get_signed_upload_url":
        "packages/functions/src/get_signed_upload_url.handler",
    },
  });
  // Show the API endpoint in the output
  stack.addOutputs({
    ApiEndpoint: api.url,
    GetSignedUploadUrlFunction: api.getFunction(
      "GET /lambda/get_signed_upload_url"
    )?.functionName,
  });
}
