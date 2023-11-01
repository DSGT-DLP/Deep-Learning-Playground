import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import parseJwt from "@dlp-sst-app/core/parseJwt";
import { v4 as uuidv4 } from 'uuid';
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, PutCommand, PutCommandInput } from '@aws-sdk/lib-dynamodb';
import { TrainStatus } from './constants';

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
    if (event) {
        const user_id: string = parseJwt(event.headers.authorization ?? "")["user_id"];
        const eventBody = JSON.parse(event.body? event.body : "");

        const trainspaceId = uuidv4();
        let putCommandInput: PutCommandInput = {
            TableName: "trainspace",
            Item:
            {
                trainspace_id: trainspaceId,
                uid: user_id,
                created: Date.now().toString(),
                data_source: 'TABULAR',
                dataset_data: JSON.stringify(eventBody['dataset_data']),
                name: eventBody['name'],
                parameters_data: { 
                    criterion: eventBody['criterion'],
                    optimizer_name: eventBody['optimizer_name'],
                    shuffle: eventBody['shuffle'],
                    epochs: eventBody['epochs'],
                    batch_size: eventBody['batch_size'],
                    user_arch: eventBody['user_arch']
                },
                review_data: "",
                status: TrainStatus.QUEUED
            }
        }

        if (putCommandInput == null)
        {
            return {
                statusCode: 400,
                body: JSON.stringify({ message: "Invalid request body" })
            }
        }

        const client = new DynamoDBClient({});
        const docClient = DynamoDBDocumentClient.from(client);

        const command = new PutCommand(putCommandInput);
        const response = await docClient.send(command);

        if (response.$metadata.httpStatusCode != 200) {
            return {
                statusCode: 500,
                body: JSON.stringify({ message: "Internal server error."})
            };
        }
        
        return {
            statusCode: 200,
            body: JSON.stringify({ trainspaceId: trainspaceId, message: "Successfully created a new trainspace."})
        };
      }
    return {
        statusCode: 404,
        body: JSON.stringify({ message: "Not Found" }),
    };
};