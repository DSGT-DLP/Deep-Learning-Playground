import { APIGatewayProxyHandlerV2, APIGatewayProxyEventV2 } from "aws-lambda";
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, GetCommand } from '@aws-sdk/lib-dynamodb';

export const handler: APIGatewayProxyHandlerV2 = async (event : APIGatewayProxyEventV2) => {
    const queryParams = event['pathParameters'];
    if (queryParams != null)
    {
        const trainspaceId: string | undefined = queryParams['id'];
        
        if (trainspaceId == undefined) {
            return {
                statusCode: 400,
                body: JSON.stringify({message: "Malformed request content - trainspace ID missing."})
            };
        }

        const client: DynamoDBClient = new DynamoDBClient({});
        const docClient = DynamoDBDocumentClient.from(client);
        
        const command : GetCommand = new GetCommand({
            TableName : "TrainspaceTable",
            Key : 
            {
                trainspace_id : trainspaceId
            }
        });

        const response = await docClient.send(command);
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