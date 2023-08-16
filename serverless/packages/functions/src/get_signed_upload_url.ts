import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import AWS from "aws-sdk";

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
  if (
    event &&
    event.queryStringParameters &&
    event.queryStringParameters.path
  ) {
    const s3 = new AWS.S3();
    // Set the expiration time for the presigned URL (e.g., 15 minutes)
    const expirationTime = 15 * 60; // 15 minutes in seconds
    const path = `${event.queryStringParameters.path}`;
    // Generate the presigned URL for a putObject operation
    const params = {
      Bucket: "dlp-upload-bucket",
      Key: path,
      Expires: expirationTime,
      ContentType: "text/plain", // Set the appropriate content type
      ACL: "private", // Set ACL as needed
    };
    const url = s3.getSignedUrl("putObject", params);
    return {
      statusCode: 200,
      body: JSON.stringify({ data: url, message: "Success" }),
    };
  }
  return {
    statusCode: 404,
    body: JSON.stringify({ error: true }),
  };
};
