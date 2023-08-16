import { Api, StackContext } from "sst/constructs";

export function AppStack({ stack }: StackContext) {
  const api = new Api(stack, "Api", {
    authorizers: {
      FirebaseAuthorizer: {
        type: "jwt",
        jwt: {
          issuer:
            "https://securetoken.google.com/deep-learning-playground-8d2ce",
          audience: ["deep-learning-playground-8d2ce"],
        },
      },
    },
    defaults: {
      authorizer: "FirebaseAuthorizer",
    },
    routes: {
      "GET /datasets/user/{type}/{filename}/presigned_upload_url":
        "packages/functions/src/datasets/user_datasets.get_dataset_presigned_upload_url_handler",
      "GET /datasets/user/{type}":
        "packages/functions/src/datasets/user_datasets.get_datasets_handler",
      "GET /datasets/user/{type}/{filename}/columns":
        "packages/functions/src/datasets/user_datasets.get_dataset_columns_handler",
    },
  });
  // Show the API endpoint in the output
  stack.addOutputs({
    ApiEndpoint: api.url,
    GetUserDatasetPresignedUploadUrlFunctionName:
      api.getFunction("POST /user_datasets/{type}/{filename}")?.functionName ??
      "",
    GetUserDatasetsFunctionName:
      api.getFunction("GET /user_datasets/{type}")?.functionName ?? "",
  });
}
