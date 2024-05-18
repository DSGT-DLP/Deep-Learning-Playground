import { APIGatewayProxyEventV2 } from "aws-lambda";
import { beforeEach, expect, it, vi} from "vitest";
import { DynamoDBClient, GetItemCommand } from '@aws-sdk/client-dynamodb';
import { mockClient } from 'aws-sdk-client-mock';
import { handler } from '../get_user';

//mocks parseJwt so that the call just returns whatever the input is
vi.mock('@dlp-sst-app/core/src/parseJwt', async () => {
  return {
      default: vi.fn().mockImplementation(input => input),
  }
})

beforeEach(async () => {
  ddbMock.reset();
})

const ddbMock = mockClient(DynamoDBClient);

it("test successful get user call", async () => {
  ddbMock.on(GetItemCommand).resolves({
    Item: { user_id: { S: 'UID' } }
  })

  //@ts-expect-error
  const event: APIGatewayProxyEventV2 =  {
    headers: {
      authorization: 'abcd',
    },
  }

  const result = await handler(event);
  expect(result.statusCode).toEqual(200);
});

it("test no existing user id", async () => {
  ddbMock.on(GetItemCommand).resolves({
    Item: undefined
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