import { APIGatewayProxyEventV2 } from "aws-lambda";
import { DynamoDBClient, DeleteItemCommand } from '@aws-sdk/client-dynamodb';
import parseJwt from "@dlp-sst-app/core/src/parseJwt";

export async function handler<APIGatewayProxyHandlerV2>(event : APIGatewayProxyEventV2) {
    if (event) {
        const user_id: string = parseJwt(event.headers.authorization ?? "")["user_id"];
        
        const client = new DynamoDBClient({});
        
        const command = new DeleteItemCommand({
            TableName : "UserTable",
            Key :
            {
                user_id: {"S": user_id}
            }
        });

        const response = await client.send(command);

        if (response.$metadata.httpStatusCode == undefined || response.$metadata.httpStatusCode != 200) 
        {
            return {
                statusCode: 404,
                body: JSON.stringify({ message : "Delete operation failed" })
            }
        }
        return {
            statusCode: 200,
            body: "Successfully deleted user with id " + user_id
        }
    }
    return {
        statusCode: 400,
        body: JSON.stringify({ message : "Malformed request content" }),
    };
};