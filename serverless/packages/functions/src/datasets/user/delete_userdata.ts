import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import { S3Client, ListObjectsV2Command } from "@aws-sdk/client-s3";
import parseJwt from "@dlp-sst-app/core/parseJwt";

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
    return {
        statusCode: 404,
        body: JSON.stringify({ message: "Test message" }),
    };
};