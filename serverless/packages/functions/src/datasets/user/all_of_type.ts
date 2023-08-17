import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import { S3Client, ListObjectsV2Command } from "@aws-sdk/client-s3";
import parseJwt from "@dlp-sst-app/core/parseJwt";

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
  if (event && event.pathParameters && event.pathParameters.type) {
    const client = new S3Client();
    const user_id: string = parseJwt(event.headers.authorization ?? "")[
      "user_id"
    ];
    const command = new ListObjectsV2Command({
      Bucket: "dlp-upload-bucket",
      Prefix: `${user_id}/${event.pathParameters.type}`,
    });
    const s3objects = await client.send(command);
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
