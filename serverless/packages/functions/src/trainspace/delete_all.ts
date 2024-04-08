import { APIGatewayProxyEventV2 } from "aws-lambda";
import parseJwt from "@dlp-sst-app/core/src/parseJwt";
import { DynamoDBClient, QueryCommand, BatchWriteItemCommand, BatchExecuteStatementCommand } from '@aws-sdk/client-dynamodb';

export async function handler<APIGatewayProxyHandlerV2>(event : APIGatewayProxyEventV2) {
    if (event) {
        const uid: string = parseJwt(event.headers.authorization ?? "")[
            "user_id"
        ];
        const client = new DynamoDBClient({});

        const foundTrainspaceIds: Array<String> = []
        const deletedTrainspaceIds: Array<string> = [];
        let toDelete = 0;
        do {
            const queryCommand: QueryCommand = new QueryCommand({
                TableName: "TrainspaceTable",
                IndexName: "user_id_index",
                KeyConditionExpression: "user_id = :uid",
                ExpressionAttributeValues: {
                    ":uid" : {"S": uid}
                },
            });

            const getResults = await client.send(queryCommand);
            
            if (getResults["Count"] !== 0 && getResults['Items']) {
                const page: Array<string | undefined> = getResults['Items'].map(trainspace => trainspace['trainspace_id'].S);
                page.forEach(id => { if (id) foundTrainspaceIds.push(id); });
            } else {
                if (deletedTrainspaceIds.length !== 0) {
                    return { 
                        statusCode: 200, 
                        body: JSON.stringify({ message: "deleted trainspaces", trainspace_ids : deletedTrainspaceIds}) 
                    };
                }
                return {
                    statusCode: 405,
                    body: JSON.stringify({ message: "no trainspaces deleted: none found associated with user"})
                }
            }

            const JSONedDelete = {
                Statement: "DELETE FROM TrainspaceTable where trainspace_id=?",
                    Parameters: [{"S": "0f70d308-7cf2-4d4f-9e11-019310714bdd"}]
            }
            const command = new BatchExecuteStatementCommand({
                Statements: [
                    JSONtest
                ],
            });
                const response = await client.send(command);
                console.log(response.Responses);

                if (response.$metadata.httpStatusCode == undefined || response.$metadata.httpStatusCode != 200) 
                {
                    return {
                        statusCode: 404,
                        body: JSON.stringify({ message : "Delete operation failed" })
                    }
                }
        
            
        } while (toDelete !== 0);
    }
    return {
        statusCode: 400,
        body: JSON.stringify({ message: "Event Not Found" }),
    };
};