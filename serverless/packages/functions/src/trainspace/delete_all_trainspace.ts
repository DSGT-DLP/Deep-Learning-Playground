import { APIGatewayProxyEventV2 } from "aws-lambda";
import parseJwt from "@dlp-sst-app/core/src/parseJwt";
import { DynamoDBClient, QueryCommand, BatchExecuteStatementCommand, BatchStatementRequest } from '@aws-sdk/client-dynamodb';

export async function handler<APIGatewayProxyHandlerV2>(event : APIGatewayProxyEventV2) {
    if (event) {
        const uid: string = parseJwt(event.headers.authorization ?? "")[
            "user_id"
        ];
        const client = new DynamoDBClient({});

        const foundTrainspaceIds: Array<String> = []
        const deletedTrainspaceIds: Array<string> = [];
        let lastEvaluatedKey = undefined;
        do {
            const queryCommand: QueryCommand = new QueryCommand({
                TableName: "TrainspaceTable",
                IndexName: "user_id_index",
                KeyConditionExpression: "user_id = :uid",
                ExpressionAttributeValues: {
                    ":uid" : {"S": uid}
                },
                Limit: 25,
                ExclusiveStartKey: lastEvaluatedKey
            });

            const currentTrainspaceIds: Array<String> = []
            const getResults = await client.send(queryCommand);
            if (getResults["Count"] !== 0 && getResults['Items']) {
                const page: Array<string | undefined> = getResults['Items'].map(trainspace => trainspace['trainspace_id'].S);
                page.forEach(id => {
                    if (id) foundTrainspaceIds.push(id); 
                    if (id) currentTrainspaceIds.push(id); 
                });
            } else {
                return {
                    statusCode: 405,
                    body: JSON.stringify({ message: "no trainspaces deleted: none found associated with user"})
                }
            }
            
            lastEvaluatedKey = getResults.LastEvaluatedKey;

            const statements: BatchStatementRequest[] = [];
            for (const id of currentTrainspaceIds) {
                statements.push( {
                    Statement: "DELETE FROM TrainspaceTable where trainspace_id=?",
                    Parameters: [{ "S": id.valueOf() }]
                });
                deletedTrainspaceIds.push(id.valueOf());
            }

            const command = new BatchExecuteStatementCommand({
                Statements: statements
            });
            
            const response = await client.send(command);

            if (response.$metadata.httpStatusCode == undefined || response.$metadata.httpStatusCode != 200) 
            {
                return {
                    statusCode: 404,
                    body: JSON.stringify({ message : "Delete operation failed" })
                }
            }
        } while (lastEvaluatedKey !== undefined);
        if (deletedTrainspaceIds.length !== 0) {
            return { 
                statusCode: 200, 
                body: JSON.stringify({ message: "Succesfully deleted trainspaces", trainspace_ids : foundTrainspaceIds}) 
            };
        }
    }
    return {
        statusCode: 400,
        body: JSON.stringify({ message: "Event Not Found" }),
    };
};