import { APIGatewayProxyEventV2 } from "aws-lambda";
import parseJwt from "@dlp-sst-app/core/src/parseJwt";
import { DynamoDBClient, QueryCommand } from '@aws-sdk/client-dynamodb';

export async function handler<APIGatewayProxyHandlerV2>(event : APIGatewayProxyEventV2) {
    if (event) {
        const uid: string = parseJwt(event.headers.authorization ?? "")[
            "user_id"
        ];
        
        const client = new DynamoDBClient({});
        const fetchedTrainspaceIds: Array<string> = [];
        let lastEvaluatedKey = undefined;
        do {
            const queryCommand: QueryCommand = new QueryCommand({
                TableName: "TrainspaceTable",
                IndexName: "user_id_index",
                KeyConditionExpression: "user_id = :uid",
                ExpressionAttributeValues: {
                    ":uid" : {"S": uid}
                },
                ExclusiveStartKey: lastEvaluatedKey
            });
            
            const results = await client.send(queryCommand); 
            
            if (results.Items && results.Count != 0) {
                const page: Array<string | undefined> = results['Items']?.map(trainspace => trainspace['trainspace_id'].S);
                page.forEach(id => { if (id) fetchedTrainspaceIds.push(id); });
            } else {
                return {
                    statusCode: 404,
                    body: JSON.stringify({message: "no trainspaces associated with user"})
                }
            }
            lastEvaluatedKey = results.LastEvaluatedKey;
        } while (lastEvaluatedKey !== undefined);
        
        return { 
            statusCode: 200, 
            body: JSON.stringify({ trainspace_ids : fetchedTrainspaceIds}) 
        };
    }

    return {
        statusCode: 400,
        body: JSON.stringify({ message: "Not Found" }),
    };
};