import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import parseJwt from "@dlp-sst-app/core/parseJwt";
import TrainspaceData from './trainspace-data';
import { v4 as uuidv4 } from 'uuid';
import { DynamoDBClient, PutItemCommand } from '@aws-sdk/client-dynamodb'; //@aws-sdk/client-dynamodb
import { DynamoDBDocumentClient } from '@aws-sdk/lib-dynamodb';

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
    if (event) {
        const uid: string = parseJwt(event.headers.authorization ?? "")["user_id"];
        console.log("Uid: " + uid);
        const trainspace_id: string = uuidv4();
        const trainspaceData: TrainspaceData = new TrainspaceData(trainspace_id, uid);

        const client: DynamoDBClient = new DynamoDBClient({});
        const docClient: DynamoDBDocumentClient = DynamoDBDocumentClient.from(client);
        
        const command: PutItemCommand = new PutItemCommand(trainspaceData.convertToDynamoItemInput("trainspace"));

        const response = await docClient.send(command);
        if (response.$metadata.httpStatusCode != 200) {
            return {
                statusCode: 500,
                body: JSON.stringify({ message: "Internal server error."})
            };
        }
        
        return {
            statusCode: 200,
            body: JSON.stringify({ trainspaceId: trainspace_id, message: "Successfully created a new trainspace."})
        };
      }
    return {
        statusCode: 404,
        body: JSON.stringify({ message: "Not Found" }),
    };
};