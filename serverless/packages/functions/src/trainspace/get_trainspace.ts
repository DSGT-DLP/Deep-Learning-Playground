import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import { S3Client, ListObjectsV2Command } from "@aws-sdk/client-s3";
import parseJwt from "@dlp-sst-app/core/parseJwt";

import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { GetCommand, DynamoDBDocumentClient } from '@aws-sdk/lib-dynamodb';

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
    const queryParams: object = event['queryStringParameters'];
    if (queryParams != null)
    {
        const uuid: string = queryParams['id'];
        const client: DynamoDBClient = new DynamoDBClient({});

        const documentClient: DynamoDBDocumentClient = DynamoDBDocumentClient.from(client);

        const response = await documentClient.get({
            TableName: "trainspace",
            Key: {
                id: uuid
            }
        });

        if (!response.Item) {
            return {
                statusCode: 200,
                body: "Trainspace id " + uuid + " does not exist."
            }
        }
        return {
            statusCode: 200,
            body: response.Item
        }
    }
    return {
        statusCode: 404,
        body: JSON.stringify({ message: "Not Found" }),
    };
};