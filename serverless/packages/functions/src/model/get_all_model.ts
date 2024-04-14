import { APIGatewayProxyEventV2 } from "aws-lambda";
import parseJwt from "@dlp-sst-app/core/src/parseJwt";
import { DynamoDBClient, QueryCommand } from '@aws-sdk/client-dynamodb';

export async function handler(event : APIGatewayProxyEventV2) {
    if (event) {
        const uid: string = parseJwt(event.headers.authorization ?? "")[
            "user_id"
        ];
        
        const client = new DynamoDBClient({});
        const fetchedModelIds: Array<string> = [];
        let lastEvaluatedKey = undefined;
        do {
            const queryCommand: QueryCommand = new QueryCommand({
                TableName: "ModelTable",
                IndexName: "user_id_index",
                KeyConditionExpression: "user_id = :uid",
                ExpressionAttributeValues: {
                    ":uid" : {"S": uid}
                },
                ExclusiveStartKey: lastEvaluatedKey
            });
            
            const results = await client.send(queryCommand); 
            
            if (results.Items && results.Count != 0) {
                const page: Array<string | undefined> = results['Items']?.map(model => model['model_id'].S);
                page.forEach(model_id => { if (model_id) fetchedModelIds.push(model_id); });
            } else {
                return {
                    statusCode: 404,
                    body: JSON.stringify({message: "no models associated with user"})
                }
            }
            lastEvaluatedKey = results.LastEvaluatedKey;
        } while (lastEvaluatedKey !== undefined);
        
        return { 
            statusCode: 200, 
            body: JSON.stringify({ model_ids : fetchedModelIds}) 
        };
    }

    return {
        statusCode: 400,
        body: JSON.stringify({ message: "Not Found" }),
    };
};