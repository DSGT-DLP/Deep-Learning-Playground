import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import parseJwt from "@dlp-sst-app/core/parseJwt";
import { DynamoDBClient, QueryCommand } from '@aws-sdk/client-dynamodb';

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
    if (event) {
        const user_id: string = parseJwt(event.headers.authorization ?? "")[
            "user_id"
        ];

        const client = new DynamoDBClient({});

        const fetchedTrainspaceIds: Array<string> = [];
        let lastEvaluatedKey = undefined;

        do {
            const getCommand: QueryCommand = new QueryCommand({
                TableName: "trainspace",
                IndexName: "uid",
                KeyConditionExpression: "uid = :uid",
                ExpressionAttributeValues: {
                    ":uid" : 
                    { 
                        S: user_id
                    }
                },
                ExclusiveStartKey: lastEvaluatedKey
            });

            const results = await client.send(getCommand);
            lastEvaluatedKey = results.LastEvaluatedKey;

            if (results['Items']) {
                const page: Array<string | undefined> = results['Items']?.map(trainspace => trainspace['trainspace_id'].S);
                page.forEach(id => { if (id) fetchedTrainspaceIds.push(id); });
            }
            

        } while (lastEvaluatedKey !== undefined);
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