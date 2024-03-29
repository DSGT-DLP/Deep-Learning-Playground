import { APIGatewayProxyHandlerV2 } from "aws-lambda";

import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, DeleteCommand } from '@aws-sdk/lib-dynamodb';

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
    let queryParams = null;
    if (event && (queryParams = event['pathParameters']) != null) {
        const trainspaceId: string | undefined = queryParams['id'];

        if (trainspaceId == undefined) {
            return {
                statusCode: 400,
                body: JSON.stringify({ message : "Malformed request content - trainspace ID missing." }),
            };
        }
        
        const client = new DynamoDBClient({});
        const docClient = DynamoDBDocumentClient.from(client);
        
        const command = new DeleteCommand({
            TableName : "trainspace",
            Key :
            {
                trainspace_id: trainspaceId
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
            body: "Successfully deleted trainspace with id " + trainspaceId
        }
    }
    return {
        statusCode: 400,
        body: JSON.stringify({ message : "Malformed request content" }),
    };
};