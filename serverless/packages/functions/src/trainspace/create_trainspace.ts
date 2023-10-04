import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import parseJwt from "@dlp-sst-app/core/parseJwt";
import TrainspaceData from './trainspace-data';
import { v4 as uuidv4 } from 'uuid';
import { DynamoDBClient, PutItemCommand } from '@aws-sdk/client-dynamodb'; //@aws-sdk/client-dynamodb
import { DynamoDBDocumentClient } from '@aws-sdk/lib-dynamodb';
import { DATA_SOURCE_ARR } from '../../../../../frontend/src/features/Train/constants/trainConstants';

function validateRequestBody(eventBody: any) : boolean {
    if (!eventBody['data_source']) {
            return false;
    }

    if (!DATA_SOURCE_ARR.includes(eventBody['data_source'])) {
        return false;
    }

    if (!eventBody['dataset_data']) {
        return false;
    }

    if (!eventBody['name']) {
        return false;
    }

    if (!eventBody['parameters_data']) {
        return false;
    }

    if (!eventBody['review_data']) {
        return false;
    }
    return true;
}

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
    if (event) {
        const uid: string = parseJwt(event.headers.authorization ?? "")["user_id"];
        const eventBody = JSON.parse(event.body? event.body : "");
        
        if (!validateRequestBody(eventBody)) {
            return {
                statusCode: 400,
                body: JSON.stringify({ message: "Invalid request body" })
            }
        }

        const trainspaceId = uuidv4();
        const trainspaceData = new TrainspaceData(trainspaceId, uid, eventBody['data_source'], eventBody['dataset_data'], eventBody['name'], eventBody['parameters_data'], eventBody['review_data']);


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