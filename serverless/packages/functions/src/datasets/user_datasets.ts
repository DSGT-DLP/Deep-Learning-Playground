import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import AWS from "aws-sdk";
import { S3Client, SelectObjectContentCommand } from "@aws-sdk/client-s3";
import parseJwt from "@dlp-sst-app/core/parseJwt";

export const get_dataset_presigned_upload_url_handler: APIGatewayProxyHandlerV2 =
  async (event) => {
    if (
      event &&
      event.pathParameters &&
      event.pathParameters.type &&
      event.pathParameters.filename
    ) {
      const s3 = new AWS.S3();
      const user_id: string = parseJwt(event.headers.authorization ?? "")[
        "user_id"
      ];
      // Generate the presigned URL for a putObject operation
      const url = s3.getSignedUrl("putObject", {
        Bucket: "dlp-upload-bucket",
        Key: `${user_id}/${event.pathParameters.type}/${event.pathParameters.filename}`,
        Expires: 15 * 60, // expiration time in seconds
        ContentType: "text/plain", // Set the appropriate content type
        ACL: "private", // Set ACL as needed
      });
      return {
        statusCode: 200,
        body: JSON.stringify({ data: url, message: "Success" }),
      };
    }
    return {
      statusCode: 404,
      body: JSON.stringify({ message: "Not Found" }),
    };
  };

export const get_dataset_columns_handler: APIGatewayProxyHandlerV2 = async (
  event
) => {
  if (
    event &&
    event.pathParameters &&
    event.pathParameters.type &&
    event.pathParameters.filename
  ) {
    const s3client = new S3Client();
    const user_id: string = parseJwt(event.headers.authorization ?? "")[
      "user_id"
    ];
    const input = {
      Bucket: "dlp-upload-bucket",
      Key: `${user_id}/${event.pathParameters.type}/${event.pathParameters.filename}`,
      Expression: "SELECT * FROM S3Object LIMIT 1",
      ExpressionType: "SQL",
      InputSerialization: {
        CSV: {
          FileHeaderInfo: "NONE",
        },
      },
      OutputSerialization: {
        CSV: {},
      },
    };
    const command = new SelectObjectContentCommand(input);
    const response = await s3client.send(command);
    if (!response.Payload) {
      return {
        statusCode: 404,
        body: JSON.stringify({ message: "Payload Not Found" }),
      };
    }
    for await (const event of response.Payload) {
      if (!event.Records?.Payload) {
        return {
          statusCode: 404,
          body: JSON.stringify({ message: "Column Names Not Found" }),
        };
      }
      const columns = new TextDecoder()
        .decode(event.Records.Payload)
        .split("\n")[0]
        .split(",");
      return {
        statusCode: 200,
        body: JSON.stringify({ data: columns, message: "Success" }),
      };
    }
  }
  return {
    statusCode: 404,
    body: JSON.stringify({ message: "Not Found" }),
  };
};

export const get_datasets_handler: APIGatewayProxyHandlerV2 = async (event) => {
  if (event && event.pathParameters && event.pathParameters.type) {
    const s3 = new AWS.S3();
    const user_id: string = parseJwt(event.headers.authorization ?? "")[
      "user_id"
    ];
    const s3objects = await s3
      .listObjectsV2({
        Bucket: "dlp-upload-bucket",
        Prefix: `${user_id}/${event.pathParameters.type}`,
      })
      .promise();
    return {
      statusCode: 200,
      body: JSON.stringify({
        data: s3objects.Contents ?? [],
        message: "Success",
      }),
    };
  }
  return {
    statusCode: 404,
    body: JSON.stringify({ message: "Not Found" }),
  };
};
