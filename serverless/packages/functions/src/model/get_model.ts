import { APIGatewayProxyEventV2 } from "aws-lambda";
import { DynamoDBClient, GetItemCommand } from '@aws-sdk/client-dynamodb';

export async function handler(event : APIGatewayProxyEventV2) {
    let queryParams = null;

    if (event && ((queryParams = event['pathParameters']) != null))
    {
        const model_id: string | undefined = queryParams['model_id'];
        if (model_id == undefined) {
            return {
                statusCode: 401,
                body: JSON.stringify({message: "Malformed request content - model ID missing."})
            }
        }

        const client: DynamoDBClient = new DynamoDBClient({});

        const command : GetItemCommand = new GetItemCommand({
            TableName : "ModelTable",
            Key : 
            {
                model_id : {"S": model_id}
            }
        });

        const response = await client.send(command);

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