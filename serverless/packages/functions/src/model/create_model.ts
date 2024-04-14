import { APIGatewayProxyEventV2 } from "aws-lambda";
import parseJwt from "@dlp-sst-app/core/src/parseJwt";
import { v4 as uuidv4 } from 'uuid';
import { DynamoDBClient, PutItemCommand } from '@aws-sdk/client-dynamodb';

export async function handler(event : APIGatewayProxyEventV2) {
    if (event) {
        const user_id: string = parseJwt(event.headers.authorization ?? "")["user_id"];
        const model_id = uuidv4();

        const client = new DynamoDBClient({});

        const eventBody = JSON.parse(event.body? event.body : "");
        const putCommand: PutItemCommand = new PutItemCommand({
            TableName: "ModelTable",
            Item:
            {
                user_id: {"S": user_id},
                model_id: {"S": model_id},
                name: {"S": eventBody['name']},
                model_structure: {"S": eventBody['model_structure']}
            }
        });

        if (putCommand == null)
        {
            return {
                statusCode: 400,
                body: JSON.stringify({ message: "Invalid request body" })
            }
        }

        const response = await client.send(putCommand);
        
        if (response.$metadata.httpStatusCode != 200) {
            return {
                statusCode: 500,
                body: JSON.stringify({ message: "Internal server error."})
            };
        }
        
        return {
            statusCode: 200,
            body: JSON.stringify({ model_id: model_id, message: "Successfully created a new model."})
        };
      }
    return {
        statusCode: 404,
        body: JSON.stringify({ message: "Not Found" }),
    };
};