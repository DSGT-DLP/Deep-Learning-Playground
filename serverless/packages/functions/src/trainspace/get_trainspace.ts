import { APIGatewayProxyEventV2 } from "aws-lambda";
import { DynamoDBClient, GetItemCommand } from '@aws-sdk/client-dynamodb';

export async function handler<APIGatewayProxyHandlerV2>(event : APIGatewayProxyEventV2) {
    let queryParams = null;
    if (event && (queryParams = event['pathParameters']) != null) {
        const trainspaceId: string | undefined = queryParams['id'];
        
        if (trainspaceId == undefined) {
            return {
                statusCode: 401,
                body: JSON.stringify({message: "Malformed request content - trainspace ID missing."})
            };
        }

        const client: DynamoDBClient = new DynamoDBClient({});
        
        const command : GetItemCommand = new GetItemCommand({
            TableName : "TrainspaceTable",
            Key : 
            {
                trainspace_id : {"S": trainspaceId}
            }
        });

        const response = await client.send(command);
        if (!response.Item)
        {
            return {
                statusCode: 404,
                body: JSON.stringify({message: "Provided trainspaceId does not exist"})
            }
        }
        return {
            statusCode: 200,
            body: JSON.stringify({message: "Successfully retrieved trainspace data", trainspace: response.Item})
        }
    }
    return {
        statusCode: 400,
        body: JSON.stringify({message: "Malformed request content"})
    };
};