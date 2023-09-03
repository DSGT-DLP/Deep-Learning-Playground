import { APIGatewayProxyHandlerV2 } from "aws-lambda";
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
        
        const uid: string = parseJwt(event.headers.authorization ?? "")["user_id"];
        const trainspace_id: string = uuidv4();
        const trainspaceData: TrainspaceData = new TrainspaceData(trainspace_id, uid);

        const client: DynamoDBClient = new DynamoDBClient({});
        const docClient: DynamoDBDocumentClient = new DynamoDBDocumentClient.from(client);
        
        const command: PutCommand = new PutCommand({
            TableName: "trainspace",
            Item: trainspaceData
        });

        const response = await docClient.send(command);
        return {
            statusCode: 200,
            body: JSON.stringify({ message: "Successfully created a new trainspace."})
        };
      }
    return {
        statusCode: 404,
        body: JSON.stringify({ message: "Not Found" }),
    };
};