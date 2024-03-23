import { APIGatewayProxyHandlerV2, APIGatewayProxyEventV2 } from "aws-lambda";
import parseJwt from "../../../core/src/parseJwt";
import { v4 as uuidv4 } from 'uuid';
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, PutCommand, PutCommandInput } from '@aws-sdk/lib-dynamodb';

export async function handler<APIGatewayProxyHandlerV2>(event : APIGatewayProxyEventV2) {
    if (event) {
        const user_id: string = parseJwt(event.headers.authorization ?? "")["user_id"];
        const model_id = uuidv4();
        const eventBody = JSON.parse(event.body? event.body : "");
        let putCommandInput: PutCommandInput = {
            TableName: "ModelTable",
            Item:
            {
                user_id: user_id,
                // model_id: eventBody['model_id'],
                model_id: model_id,
                name: eventBody['name'],
                model_structure: eventBody['model_structure']
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
            body: JSON.stringify({ user_id: user_id, message: "Successfully created a new model."})
        };
      }
    return {
        statusCode: 404,
        body: JSON.stringify({ message: "Not Found" }),
    };
};
function removeUndefinedValues(obj: { [key: string]: any }) {
    const newObj: { [key: string]: any } = {};
    for (const key in obj) {
        if (obj[key] !== undefined) {
            newObj[key] = obj[key];
        }
    }
    return newObj;
}