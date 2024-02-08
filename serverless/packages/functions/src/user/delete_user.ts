import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, DeleteCommand } from '@aws-sdk/lib-dynamodb';
import parseJwt from "@dlp-sst-app/core/parseJwt";

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
    if (event) {
        const user_id: string = parseJwt(event.headers.authorization ?? "")["user_id"];
        
        const client = new DynamoDBClient({});
        const docClient = DynamoDBDocumentClient.from(client);
        
        const command = new DeleteCommand({
            TableName : "UserTable",
            Key :
            {
                user_id: user_id
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
            body: "Successfully deleted user with id " + user_id
        }
    }
    return {
        statusCode: 400,
        body: JSON.stringify({ message : "Malformed request content" }),
    };
};