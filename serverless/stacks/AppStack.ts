import { Api, Bucket, StackContext, Table } from "sst/constructs";
import { Bucket as s3Bucket } from "aws-cdk-lib/aws-s3";
import { table_exists } from "../packages/functions/src/dbutils/table_exists";

export function AppStack({ stack }: StackContext) {
  const bucket = new Bucket(stack, "dlp-upload-bucket", {
    cdk: {
      bucket: s3Bucket.fromBucketArn(
        stack,
        "i-dlp-upload-bucket",
        "arn:aws:s3:::dlp-upload-bucket"
      ),
    },
  });
  let trainspaceRunTable = null;
  if (!table_exists("trainspace")) {
    trainspaceRunTable = new Table(stack, "trainspace-run", {
      fields: {
        runId: "string",
        trainspaceId: "string",
        userId: "string",
        timestamp: "string",
        resultCsvUri: "string",
        modelPtUri: "string",
        onnxUri: "string",
        confusionMatrixUri: "string",
        aucRocUri: "string"
      },
      primaryIndex: {
        partitionKey: "runId",
      },
      globalIndexes: {
        "TrainspaceIndex": { partitionKey: "trainspaceId", sortKey: "timestamp"},
        "UserIndex": {partitionKey: "userId"}
      },
    });
  }
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
      function: {
        bind: [bucket],
      },
      authorizer: "FirebaseAuthorizer",
    },
    routes: {
      "GET /datasets/user/{type}/{filename}/presigned_upload_url":
        "packages/functions/src/datasets/user/upload_url.handler",
      "GET /datasets/user/{type}":
        "packages/functions/src/datasets/user/all_of_type.handler",
      "GET /datasets/user/{type}/{filename}/columns":
        "packages/functions/src/datasets/user/columns.handler",
    },
  });

  // Show the API endpoint in the output
  stack.addOutputs({
    ApiEndpoint: api.url,
    GetUserDatasetPresignedUploadUrlFunctionName:
      api.getFunction(
        "GET /datasets/user/{type}/{filename}/presigned_upload_url"
      )?.functionName ?? "",
    GetUserDatasetsFunctionName:
      api.getFunction("GET /datasets/user/{type}")?.functionName ?? "",
    GetUserDatasetColumnsFunctionName:
      api.getFunction("GET /datasets/user/{type}/{filename}/columns")
        ?.functionName ?? "",
    TrainspaceRunTableName: {
      value: "trainspace"
    }
  });
}
