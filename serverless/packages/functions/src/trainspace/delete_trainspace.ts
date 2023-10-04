import { APIGatewayProxyHandlerV2 } from "aws-lambda";

import { DynamoDBClient, DeleteItemCommand } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient } from '@aws-sdk/lib-dynamodb';

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
    let queryParams = null;
    if (event && (queryParams = event['pathParameters']) != null) {
        const trainspace_id: string | undefined = queryParams['id'];

        if (trainspace_id == undefined) {
            return {
                statusCode: 400,
                body: JSON.stringify({ message : "Malformed request content - trainspace ID missing." }),
            };
        }
        
        const client = new DynamoDBClient({});
        const documentClient = DynamoDBDocumentClient.from(client);
        
        const command = new DeleteItemCommand({
            TableName : "trainspace",
            Key :
            {
                "trainspace_id" :
                {
                     S: trainspace_id ? trainspace_id : ""
                }
            }
        });

        const response = await documentClient.send(command);
        if (response.$metadata.httpStatusCode == undefined || response.$metadata.httpStatusCode != 200) 
        {
            return {
                statusCode: 404,
                body: JSON.stringify({ message : "Delete operation failed" })
            }
        }
        return {
            statusCode: 200,
            body: "Successfully deleted trainspace with id " + trainspace_id
        }
    }
    return {
        statusCode: 400,
        body: JSON.stringify({ message : "Malformed request content" }),
    };
};