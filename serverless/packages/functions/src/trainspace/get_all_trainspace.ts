import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import parseJwt from "@dlp-sst-app/core/parseJwt";
import { DynamoDBClient, GetItemCommand } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient } from '@aws-sdk/lib-dynamodb';

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
    if (event) {
        const user_id: string = parseJwt(event.headers.authorization ?? "")[
            "user_id"
        ];
        
        const client: DynamoDBClient = new DynamoDBClient({});
        const docClient: DynamoDBDocumentClient = DynamoDBDocumentClient.from(client);

        const getCommand = new GetItemCommand({
            TableName: "uid",
            Key: {
                uid: 
                {
                    S: user_id
                }
            }
        });
        const results = await docClient.send(getCommand);
        return { statusCode: 200, body: results };
    }

    return {
        statusCode: 404,
        body: JSON.stringify({ message: "Not Found" }),
    };
};