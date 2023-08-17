import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";
import parseJwt from "@dlp-sst-app/core/parseJwt";

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
  if (
    event &&
    event.pathParameters &&
    event.pathParameters.type &&
    event.pathParameters.filename
  ) {
    const client = new S3Client();
    const user_id: string = parseJwt(event.headers.authorization ?? "")[
      "user_id"
    ];
    // Generate the presigned URL for a putObject operation
    const command = new PutObjectCommand({
      Bucket: "dlp-upload-bucket",
      ACL: "private",
      Key: `${user_id}/${event.pathParameters.type}/${event.pathParameters.filename}`,
    });
    const url = await getSignedUrl(client, command, {
      expiresIn: 15 * 60,
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
