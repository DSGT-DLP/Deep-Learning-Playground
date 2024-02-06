import { Api, Bucket, StackContext } from "sst/constructs";
import { Bucket as s3Bucket } from "aws-cdk-lib/aws-s3";

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
      "DELETE /dataset/user/{type}/{filename}" :
        "packages/functions/src/datasets/user/delete_url.handler",
      "POST /trainspace/tabular": {
        function: {
          handler: "packages/functions/src/trainspace/create_tabular_trainspace.handler",
          permissions: ["dynamodb:PutItem"]
        }
      },
      "POST /trainspace/image": {
        function: {
          handler: "packages/functions/src/trainspace/create_image_trainspace.handler",
          permissions: ["dynamodb:PutItem"]
        }
      },
      "POST /trainspace/create": {
        function: {
          handler: "packages/functions/src/trainspace/create_trainspace.handler",
          permissions: ["dynamodb:PutItem"]
        }
      },
      "GET /trainspace/{id}": {
        function: {
          handler: "packages/functions/src/trainspace/get_trainspace.handler",
          permissions: ["dynamodb:GetItem"]
        }
      },
      "GET /trainspace": {
        function: {
          handler: "packages/functions/src/trainspace/get_all_trainspace.handler",
          permissions: ["dynamodb:Query"]
        }
      },
      "DELETE /trainspace/{id}": {
        function: {
          handler: "packages/functions/src/trainspace/delete_trainspace.handler",
          permissions: ["dynamodb:DeleteItem"]
        }
      },
      "POST /user": {
        function: {
          handler: "packages/functions/src/user/create_user.handler",
          permissions: ["dynamodb:PutItem"]
        }
      }
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
    CreateTrainspaceFunctionName:
        api.getFunction("POST /trainspace/create")?.functionName ?? "",
    PutTabularTrainspaceFunctionName:
        api.getFunction("POST /trainspace/tabular")?.functionName ?? "",
    PutImageTrainspaceFunctionName:
        api.getFunction("POST /trainspace/tabular")?.functionName ?? "",
    GetAllTrainspaceIdsFunctionName:
        api.getFunction("GET /trainspace")?.functionName ?? "",
    GetTrainspaceByIdFunctionName:
        api.getFunction("GET /trainspace/{id}")?.functionName ?? "",
    DeleteTrainspaceByIdFunctionName:
        api.getFunction("DELETE /trainspace/{id}")?.functionName ?? "",
    CreateUserFunctionName:
        api.getFunction("POST /user")?.functionName ?? ""
  });
}
