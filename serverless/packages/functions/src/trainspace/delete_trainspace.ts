import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import { S3Client, ListObjectsV2Command } from "@aws-sdk/client-s3";
import parseJwt from "@dlp-sst-app/core/parseJwt";

import { DynamoDBClient, DeleteItemCommand } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient } from '@aws-sdk/lib-dynamodb';

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
    const queryParams: object = event['queryStringParameters'];
    if (queryParams != null)
    {
        const id : string = queryParams['id'];

        const client: DynamoDBClient = new DynamoDBClient({});
        const documentClient: DynamoDBDocumentClient = DynamoDBDocumentClient.from(client);
        
        const command : DeleteItemCommand = new DeleteItemCommand({
            TableName : "trainspace",
            Key :
            {
                trainspace_id : id
            }
        });

        const response = await documentClient.send(command);
        return response;
    }
    return {
        statusCode: 400,
        body: JSON.stringify({ message: "Malformed request content" }),
    };
};