import { APIGatewayProxyEventV2 } from "aws-lambda";
import { DynamoDBClient, GetItemCommand } from '@aws-sdk/client-dynamodb';
import parseJwt from "@dlp-sst-app/core/src/parseJwt";

export async function handler<APIGatewayProxyHandlerV2>(event : APIGatewayProxyEventV2) {
    if (event)
    {
        const user_id: string = parseJwt(event.headers.authorization ?? "")["user_id"];
        const client: DynamoDBClient = new DynamoDBClient({});
        
        const command : GetItemCommand = new GetItemCommand({
            TableName : "UserTable",
            Key : 
            {
                user_id : {"S": user_id}
            }
        });

        const response = await client.send(command);

        if (!response.Item)
        {
            return {
                statusCode: 404,
                body: JSON.stringify({message: "Provided User ID does not exist"})
            }
        }
        return {
            statusCode: 200,
            body: JSON.stringify({message: "Successfully retrieved User data", user: response.Item})
        }
    }
    return {
        statusCode: 400,
        body: JSON.stringify({message: "Malformed request content"})
    };
}