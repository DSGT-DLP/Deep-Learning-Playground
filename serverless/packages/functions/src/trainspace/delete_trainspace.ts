import { APIGatewayProxyEventV2 } from "aws-lambda";
import { DynamoDBClient, DeleteItemCommand } from '@aws-sdk/client-dynamodb';

export async function handler<APIGatewayProxyHandlerV2>(event : APIGatewayProxyEventV2) {
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

        const command = new DeleteItemCommand({
            TableName : "TrainspaceTable",
            Key :
            {
                trainspace_id: {"S": trainspaceId}
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
            body: "Successfully deleted trainspace with id " + trainspaceId
        }
    }
    return {
        statusCode: 400,
        body: JSON.stringify({ message : "Malformed request content" }),
    };
};