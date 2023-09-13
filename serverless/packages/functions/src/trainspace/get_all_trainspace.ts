import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import parseJwt from "@dlp-sst-app/core/parseJwt";
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient } from '@aws-sdk/lib-dynamodb';

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
    if (
        event &&
        event.pathParameters &&
        event.pathParameters.type &&
        event.pathParameters.filename
    ) {
        const user_id: string = parseJwt(event.headers.authorization ?? "")[
            "user_id"
        ];
        
        const client: DynamoDBClient = new DynamoDBClient({});
        const docClient: DynamoDBDocumentClient = DynamoDBDocumentClient.from(client);

        const params = {
            IndexName: 'uid',
            KeyConditionExpression: '#uid = :uid',
            ExpressionAttributeNames: {
                '#uid': 'uid'
            },
            ExpressionAttributeValues: {
                ':uid': user_id
            },
            TableName: 'trainspace'
        }
        const data = await docClient.query(params).promise();
        return data;
    }

    return {
        statusCode: 404,
        body: JSON.stringify({ message: "Not Found" }),
    };
};