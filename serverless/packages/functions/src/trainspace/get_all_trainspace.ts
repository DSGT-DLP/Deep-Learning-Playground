import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import parseJwt from "@dlp-sst-app/core/parseJwt";
import { DynamoDBClient, QueryCommand } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient } from '@aws-sdk/lib-dynamodb';

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
    if (event) {
        const user_id: string = parseJwt(event.headers.authorization ?? "")[
            "user_id"
        ];

        const client: DynamoDBClient = new DynamoDBClient({});
        const docClient: DynamoDBDocumentClient = DynamoDBDocumentClient.from(client);
        
        const getCommand = new QueryCommand({
            TableName: "trainspace",
            IndexName: "uid",
            KeyConditionExpression: "uid = :uid",
            ExpressionAttributeValues: {
                ":uid" : 
                { 
                    S: user_id
                }
            }
        });

        const results = await docClient.send(getCommand);
        const fetchedTrainspaceIds = results['Items']?.map(trainspace => trainspace['trainspace_id'].S);
        return { 
            statusCode: 200, 
            body: JSON.stringify({ trainspace_ids : fetchedTrainspaceIds}) 
        };
    }

    return {
        statusCode: 404,
        body: JSON.stringify({ message: "Not Found" }),
    };
};