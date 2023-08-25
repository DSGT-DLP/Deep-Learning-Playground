import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import { S3Client, SelectObjectContentCommand } from "@aws-sdk/client-s3";
import parseJwt from "@dlp-sst-app/core/parseJwt";

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
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
