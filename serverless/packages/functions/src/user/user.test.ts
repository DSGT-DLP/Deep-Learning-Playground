import { APIGatewayProxyHandlerV2, APIGatewayProxyEventV2 } from "aws-lambda";
import { beforeEach, expect, it, vi} from "vitest";
import {DynamoDBClient} from '@aws-sdk/client-dynamodb';
import {DynamoDBDocumentClient, GetCommand, QueryCommand } from '@aws-sdk/lib-dynamodb';
import parseJwt from "../../../core/src/parseJwt";
import {mockClient} from 'aws-sdk-client-mock';
import { handler } from './get_user';
import { NativeAttributeValue } from "@aws-sdk/util-dynamodb";
import 'aws-sdk-client-mock-vitest';



vi.mock('../../../core/src/parseJwt', async () => {
  return {
      default: vi.fn().mockImplementation(input => input),
  }
})

beforeEach(async () => {
  ddbMock.reset();
})

const ddbMock = mockClient(DynamoDBDocumentClient);

const dynamodb = new DynamoDBClient({});
const ddb = DynamoDBDocumentClient.from(dynamodb);

const key: Record<string, NativeAttributeValue> = {
// Specify the key attributes with their respective values
"partitionKey": { S: "UID" }, 
};

const get = await ddb.send(new GetCommand({
TableName: 'UserTable',
Key: key
}));

it("test successful get user call", async () => {
  ddbMock.on(GetCommand).resolves({
    Item: { user_id: { S: 'UID' } }
  })

  //error is fine, doesn't affect functionality. We don't need the rest of the event, and it's really long for no reason
  const event: APIGatewayProxyEventV2 =  {
    headers: {
      authorization: 'abcd',
    },
}
    
  const result = await handler(event);
  expect(result.statusCode).toEqual(200);
});

it("test no existing user id get user call", async () => {
  ddbMock.on(GetCommand).resolves({
    Item: undefined
  })

  //error is fine, doesn't affect functionality. We don't need the rest of the event, and it's really long for no reason
  const event: APIGatewayProxyEventV2 =  {
    headers: {
      authorization: 'abcd',
    },
}
    
  const result = await handler(event);
  expect(result.statusCode).toEqual(404);
});

it("test successful get user call", async () => {

  //error is fine, doesn't affect functionality. We don't need the rest of the event, and it's really long for no reason
  const event: APIGatewayProxyEventV2 =  {
    headers: {
      authorization: 'abcd',
    },
}
    
  const result = await handler(undefined);
  expect(result.statusCode).toEqual(400);
});