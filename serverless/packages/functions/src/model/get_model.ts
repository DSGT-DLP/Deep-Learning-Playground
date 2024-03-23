import { APIGatewayProxyHandlerV2, APIGatewayProxyEventV2 } from "aws-lambda";
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, GetCommand } from '@aws-sdk/lib-dynamodb';
import parseJwt from "../../../core/src/parseJwt";

export async function handler<APIGatewayProxyHandlerV2>(event : APIGatewayProxyEventV2) {
    if (event)
    {
        const model_id: string = parseJwt(event.headers.authorization ?? "")["model_id"];
        const client: DynamoDBClient = new DynamoDBClient({});
        const docClient = DynamoDBDocumentClient.from(client);
        
        const command : GetCommand = new GetCommand({
            TableName : "ModelTable",
            Key : 
            {
                model_id : model_id
            }
        });

        const response = await docClient.send(command);

        if (!response.Item)
        {
            return {
                statusCode: 404,
                body: JSON.stringify({message: "Provided Model ID does not exist"})
            }
        }
        return {
            statusCode: 200,
            body: JSON.stringify({message: "Successfully retrieved Model data", model: response.Item})
        }
    }
    return {
        statusCode: 400,
        body: JSON.stringify({message: "Malformed request content"})
    };
}