import { APIGatewayProxyEventV2 } from "aws-lambda";
import { DynamoDBClient, DeleteItemCommand } from '@aws-sdk/client-dynamodb';

export async function handler(event : APIGatewayProxyEventV2) {
    let queryParams = null;
    if (event && (queryParams = event['pathParameters']) != null) {
        const model_id: string | undefined = queryParams['model_id'];
        if (model_id == undefined) {
            return {
                statusCode: 401,
                body: JSON.stringify({message: "Malformed request content - model ID missing."})
            }
        }
        
        const client = new DynamoDBClient({});

        const command = new DeleteItemCommand({
            TableName : "ModelTable",
            Key :
            {
                model_id: {"S": model_id}
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
            body: "Successfully deleted model with id " + model_id
        }
    }
    return {
        statusCode: 400,
        body: JSON.stringify({ message : "Malformed request content" }),
    };
};