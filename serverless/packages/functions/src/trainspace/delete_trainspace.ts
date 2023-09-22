import { APIGatewayProxyHandlerV2 } from "aws-lambda";

import { DynamoDBClient, DeleteItemCommand } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient } from '@aws-sdk/lib-dynamodb';

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
    let queryParams = null;
    if (event && (queryParams = event['pathParameters']) != null) {
        const trainspace_id : string = queryParams['id'];
        console.log("trainspace_id: " + trainspace_id);
        const client: DynamoDBClient = new DynamoDBClient({});
        const documentClient: DynamoDBDocumentClient = DynamoDBDocumentClient.from(client);
        
        const command : DeleteItemCommand = new DeleteItemCommand({
            TableName : "trainspace",
            Key :
            {
                "trainspace_id" :
                {
                     S: trainspace_id
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