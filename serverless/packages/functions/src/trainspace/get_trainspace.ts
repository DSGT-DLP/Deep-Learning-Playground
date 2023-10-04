import { APIGatewayProxyHandlerV2, APIGatewayProxyEvent } from "aws-lambda";
import { DynamoDBClient, GetItemCommand } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient } from '@aws-sdk/lib-dynamodb';

export const handler: APIGatewayProxyHandlerV2 = async (event : APIGatewayProxyEvent) => {
    const queryParams = event['pathParameters'];
    if (queryParams != null)
    {
        const trainspace_id: string | undefined = queryParams['id'];
        
        if (trainspace_id == undefined) {
            return {
                statusCode: 400,
                body: JSON.stringify({message: "Malformed request content - trainspace ID missing."})
            };
        }

        const client: DynamoDBClient = new DynamoDBClient({});
        const documentClient: DynamoDBDocumentClient = DynamoDBDocumentClient.from(client);

        const command : GetItemCommand = new GetItemCommand({
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