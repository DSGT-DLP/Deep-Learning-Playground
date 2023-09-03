import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import { S3Client, ListObjectsV2Command } from "@aws-sdk/client-s3";
import parseJwt from "@dlp-sst-app/core/parseJwt";
import TrainspaceData from './trainspace-data';
import { v4 as uuidv4 } from 'uuid';
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { PutCommand, DynamoDBDocumentClient } from '@aws-sdk/lib-dynamodb';

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
    if (
        event &&
        event.pathParameters &&
        event.pathParameters.type &&
        event.pathParameters.filename
      ) {
        
        const uid = parseJwt(event.headers.authorization ?? "")["user_id"];
        const trainspace_id = uuidv4();
        const trainspaceData = new TrainspaceData(trainspace_id, uid);

        const client = new DynamoDBClient({});
        const docClient = new DynamoDBDocumentClient.from(client);
        
        const command = new PutCommand({
            TableName: "trainspace",
            Item: trainspaceData
        });

        const response = await docClient.send(command);

        return response;
      }
    return {
        statusCode: 404,
        body: JSON.stringify({ message: "Not Found" }),
    };
};