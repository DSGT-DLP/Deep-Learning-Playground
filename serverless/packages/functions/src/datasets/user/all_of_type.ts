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
    const data = (s3objects.Contents ?? []).map((obj) => {
      const [_, type, name] = (obj.Key ?? "").split("/");
      return {
        name: name,
        size: obj.Size!,
        last_modified: obj.LastModified!,
        type: type,
      };
    });
    return {
      statusCode: 200,
      body: JSON.stringify({
        data: data,
        message: "Success",
      }),
    };
  }
  return {
    statusCode: 404,
    body: JSON.stringify({ message: "Not Found" }),
  };
};
