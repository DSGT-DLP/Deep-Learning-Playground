import { APIGatewayProxyHandlerV2, APIGatewayProxyEventV2 } from "aws-lambda";
import { beforeEach, expect, it, vi} from "vitest";
import {DynamoDBClient} from '@aws-sdk/client-dynamodb';
import {DeleteCommand, DynamoDBDocumentClient, GetCommand, PutCommand } from '@aws-sdk/lib-dynamodb';
import {mockClient} from 'aws-sdk-client-mock';
import { handler } from '../get_model';
import 'aws-sdk-client-mock-vitest';

//mocks parseJwt so that the call just returns whatever the input is
vi.mock('../../../../core/src/parseJwt', async () => {
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

it("test successful get model call", async () => {
  ddbMock.on(GetCommand).resolves({
    Item: { model_id: { S: 'MID' } }
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

it("test no existing model id", async () => {
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

it("test malformed request", async () => {
    
  const result = await handler(undefined);
  expect(result.statusCode).toEqual(400);
});