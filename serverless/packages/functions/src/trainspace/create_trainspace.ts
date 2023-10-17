import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import parseJwt from "@dlp-sst-app/core/parseJwt";
import TrainspaceData from './trainspace-data';
import { v4 as uuidv4 } from 'uuid';
import { DynamoDBClient, PutItemCommand } from '@aws-sdk/client-dynamodb'; //@aws-sdk/client-dynamodb
import { DynamoDBDocumentClient } from '@aws-sdk/lib-dynamodb';

function validateRequestBody(eventBody: any, trainspace_id: string, uid: string) : TrainspaceData | null {
    console.log(eventBody);
    
    switch(eventBody['data_source'])
    {
        case "TABULAR":
            return new TrainspaceData(trainspace_id, uid, "TABULAR", eventBody['default'], eventBody['name'], 
                { 
                    criterion: eventBody['criterion'],
                    optimizer_name: eventBody['optimizer_name'],
                    shuffle: eventBody['shuffle'],
                    epochs: eventBody['epochs'],
                    batch_size: eventBody['batch_size'],
                    user_arch: eventBody['user_arch']
                }, "");
        case "PRETRAINED":

            return null;
        case "IMAGE":
            return new TrainspaceData(trainspace_id, uid, "IMAGE", eventBody['dataset_data'], eventBody['name'], eventBody['parameters_data'], eventBody['review_data']['notification_email']);
        case "AUDIO":

            return null;
        case "TEXTUAL":

            return null;
        case "CLASSICAL_ML":

            return null;
        case "OBJECT_DETECTION":

            return null;
        default:
            return null;
    }
}

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
    if (event) {
        const uid: string = parseJwt(event.headers.authorization ?? "")["user_id"];
        const eventBody = JSON.parse(event.body? event.body : "");

        const trainspaceId = uuidv4();
        const trainspaceData = validateRequestBody(eventBody);

        if (trainspaceData == null)
        {
            return {
                statusCode: 400,
                body: JSON.stringify({ message: "Invalid request body" })
            }
        }


        const client = new DynamoDBClient({});
        const docClient = DynamoDBDocumentClient.from(client);
        
        const command = new PutItemCommand(trainspaceData.convertToDynamoItemInput("trainspace"));

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