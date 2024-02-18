import { APIGatewayProxyHandlerV2, APIGatewayProxyEventV2 } from "aws-lambda";
import { beforeEach, expect, it, vi} from "vitest";
import {DynamoDBClient} from '@aws-sdk/client-dynamodb';
import {DynamoDBDocumentClient, GetCommand, QueryCommand} from '@aws-sdk/lib-dynamodb';
import {mockClient} from 'aws-sdk-client-mock';
import { handler } from "@dlp-sst-app/functions/user/get_user";
import { NativeAttributeValue } from "@aws-sdk/util-dynamodb";
import 'aws-sdk-client-mock-vitest';

//just run pnpm test to see the issue, trying to fix the handler import

beforeEach(async () => {
  ddbMock.reset();
})

const ddbMock = mockClient(DynamoDBDocumentClient);
ddbMock.on(GetCommand).resolves({
    Item: [{user_id: 'Seth123', email: 'mock@gmail.com', name: "Seth", phone: "012-345-6789"}],
});

const dynamodb = new DynamoDBClient({});
const ddb = DynamoDBDocumentClient.from(dynamodb);

const key: Record<string, NativeAttributeValue> = {
  // Specify the key attributes with their respective values
  "partitionKey": { S: "user_id" },  // Example: String attribute
};

const get = await ddb.send(new GetCommand({
  TableName: 'UserTable',
  Key: key
}));


it("test successful get user call", async () => {
  ddbMock.on(GetCommand).resolves({
    Item: { user_id: { S: 'Seth123' } }
  })

  const event: APIGatewayProxyEventV2 = {
    version: '2.0',
    routeKey: '$default',
    rawPath: '/',
    rawQueryString: '',
    headers: {
      authorization: 'Bearer <sample token>'
    },
    body: JSON.stringify({ user_id: 'Seth123' }),
    isBase64Encoded: false,
    requestContext: {
      accountId: '123456789012',
      apiId: 'api-id',
      domainName: 'testDomainName',
      domainPrefix: 'testPrefix',
      http: {
        method: 'GET',
        path: '/',
        protocol: 'HTTP/1.1',
        sourceIp: 'IP',
        userAgent: 'User-Agent'
      },
      requestId: 'id',
      routeKey: '$default',
      stage: 'stage',
      time: 'time',
      timeEpoch: 0
    }
  };

  const result = await handler(event);

  console.log(result)
});

/**
 * goal for unit test: test cases for the endpoint
 * 
 * goal: called the handler so that we can mock certain parts of the implementation of the handler
 * 
 * we want to hit the endpoints without hitting the actual table
 * 
 * figure out how to call handler using not the real dynamo table. We want to say if the fetch was successful, give this status code
 *      not depend on the actual database, but we want to use the function which hits the database
 * 
 * basically we want to make a mock of the dynamo table and pass in differents type of get user requests (failing, passing, unauthorized)
 *      to check that when we give the endpoint certain information, we get the expected results
 * 
 * we don't know if it's possible to call the real handler without calling the real database
 * 
 * idea:
 * mock the DynamoDBClient, DynamoDBDocumentClient, and GetCommand
 * 
 * steps:
 * figure out format for the input
 */