import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import parseJwt from "@dlp-sst-app/core/parseJwt";
import { v4 as uuidv4 } from 'uuid';
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, PutCommand } from '@aws-sdk/lib-dynamodb';
import { create_trainspace } from '../dbutils/put_trainspace';

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
    if (event) {
        const uid: string = parseJwt(event.headers.authorization ?? "")["user_id"];
        const eventBody = JSON.parse(event.body? event.body : "");

        const trainspaceId = uuidv4();
        const putCommandInput = create_trainspace(trainspaceId, uid, "TABULAR", eventBody['default'], eventBody['name'], 
            { 
                criterion: eventBody['criterion'],
                optimizer_name: eventBody['optimizer_name'],
                shuffle: eventBody['shuffle'],
                epochs: eventBody['epochs'],
                batch_size: eventBody['batch_size'],
                user_arch: eventBody['user_arch']
            }, "");

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