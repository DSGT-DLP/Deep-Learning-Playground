import { APIGatewayProxyHandlerV2, APIGatewayProxyEvent } from "aws-lambda";
import { S3Client, ListObjectsV2Command } from "@aws-sdk/client-s3";
import parseJwt from "@dlp-sst-app/core/parseJwt";

import { DynamoDBClient, GetItemCommand } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient } from '@aws-sdk/lib-dynamodb';

export const handler: APIGatewayProxyHandlerV2 = async (event : APIGatewayProxyEvent) => {
    const queryParams: object = event['queryStringParameters'];
    if (queryParams != null)
    {
        const id: string = queryParams['id'];

        const client: DynamoDBClient = new DynamoDBClient({});
        const documentClient: DynamoDBDocumentClient = DynamoDBDocumentClient.from(client);

        const command : GetItemCommand = new GetItemCommand({
            TableName : "trainspace",
            Key : 
            {
                trainspace_id : id
            }
        });

        const response = await documentClient.get(command);

        if (!response.Item) {
            return {
                statusCode: 400,
                body: "Trainspace id " + id + " does not exist."
            }
        }
        return {
            statusCode: 200,
            body: response.Item
        }
    }
    return {
        statusCode: 404,
        body: JSON.stringify({ message: "Malformed request content" }),
    };
};