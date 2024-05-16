import { APIGatewayProxyEventV2 } from "aws-lambda";
import parseJwt from "@dlp-sst-app/core/src/parseJwt";
import { DynamoDBClient, QueryCommand, BatchExecuteStatementCommand, BatchStatementRequest } from '@aws-sdk/client-dynamodb';

export async function handler(event : APIGatewayProxyEventV2) {
    if (event) {
        const uid: string = parseJwt(event.headers.authorization ?? "")[
            "user_id"
        ];
        const client = new DynamoDBClient({});

        const foundModelIds: Array<String> = []
        const deletedModelIds: Array<string> = [];
        let lastEvaluatedKey = undefined;
        do {
            const queryCommand: QueryCommand = new QueryCommand({
                TableName: "ModelTable",
                IndexName: "user_id_index",
                KeyConditionExpression: "user_id = :uid",
                ExpressionAttributeValues: {
                    ":uid" : {"S": uid}
                },
                Limit: 25,
                ExclusiveStartKey: lastEvaluatedKey
            });

            const currentModelIds: Array<String> = []
            const getResults = await client.send(queryCommand);
            if (getResults["Count"] !== 0 && getResults['Items']) {
                const page: Array<string | undefined> = getResults['Items'].map(model => model['model_id'].S);
                page.forEach(model_id => {
                    if (model_id) foundModelIds.push(model_id); 
                    if (model_id) currentModelIds.push(model_id); 
                });
            } else {
                return {
                    statusCode: 405,
                    body: JSON.stringify({ message: "no models deleted: none found associated with user"})
                }
            }
            
            lastEvaluatedKey = getResults.LastEvaluatedKey;

            const statements: BatchStatementRequest[] = [];
            for (const model_id of currentModelIds) {
                statements.push( {
                    Statement: "DELETE FROM ModelTable where model_id=?",
                    Parameters: [{ "S": model_id.valueOf() }]
                });
                deletedModelIds.push(model_id.valueOf());
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
        if (deletedModelIds.length !== 0) {
            return { 
                statusCode: 200, 
                body: JSON.stringify({ message: "Succesfully deleted models", model_ids : foundModelIds}) 
            };
        }
    }
    return {
        statusCode: 400,
        body: JSON.stringify({ message: "Event Not Found" }),
    };
};