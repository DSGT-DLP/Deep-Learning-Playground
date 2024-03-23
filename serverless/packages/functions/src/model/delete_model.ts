import { APIGatewayProxyHandlerV2, APIGatewayProxyEventV2 } from "aws-lambda";
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, DeleteCommand } from '@aws-sdk/lib-dynamodb';
import parseJwt from "../../../core/src/parseJwt";

export async function handler<APIGatewayProxyHandlerV2>(event : APIGatewayProxyEventV2) {
    if (event) {
        const model_id: string = parseJwt(event.headers.authorization ?? "")["model_id"];
        
        const client = new DynamoDBClient({});
        const docClient = DynamoDBDocumentClient.from(client);
        
        const command = new DeleteCommand({
            TableName : "ModelTable",
            Key :
            {
                model_id: model_id
            }
        });

        const response = await docClient.send(command);

        if (response.$metadata.httpStatusCode == undefined || response.$metadata.httpStatusCode != 200) 
        {
            return {
                statusCode: 404,
                body: JSON.stringify({ message : "Delete operation failed" })
            }
        }
        return {
            statusCode: 200,
            body: "Successfully deleted model with id " + model_id
        }
    }
    return {
        statusCode: 400,
        body: JSON.stringify({ message : "Malformed request content" }),
    };
};