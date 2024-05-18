import { APIGatewayProxyEventV2 } from "aws-lambda";
import { beforeEach, expect, it, vi} from "vitest";
import { mockClient } from 'aws-sdk-client-mock';
import { DynamoDBClient, QueryCommand } from '@aws-sdk/client-dynamodb';
import { handler } from '../get_all_trainspace';

//mocks parseJwt so that the call just returns whatever the input is
vi.mock('@dlp-sst-app/core/src/parseJwt', async () => {
  return {
    default: (input: String) => ({ user_id: input })
  }
})

beforeEach(async () => {
  ddbMock.reset();
})

const ddbMock = mockClient(DynamoDBClient);

it("test successful get all trainspace call", async () => {
  ddbMock.on(QueryCommand).resolves({
    "Items": [{
      "user_id":{"S":"abcd"},
      "trainspace_id": { "S": "test id" },
    }],
    "Count": 4
  });
  //@ts-expect-error
  const event: APIGatewayProxyEventV2 =  {
    headers: {
      authorization: 'abcd',
    },
  }

  const result = await handler(event);

  expect(result.statusCode).toEqual(200);
});

it("test no existing trainspaces for user id", async () => {
  ddbMock.on(QueryCommand).resolves({
    Items: undefined
  })

  //@ts-expect-error
  const event: APIGatewayProxyEventV2 =  {
    headers: {
      authorization: 'abcd',
    },
  }
    
  const result = await handler(event);
  expect(result.statusCode).toEqual(404);
});

it("test malformed request", async () => {
  //@ts-expect-error
  const result = await handler(undefined);
  expect(result.statusCode).toEqual(400);
});