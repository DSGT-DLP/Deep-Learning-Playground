import { APIGatewayProxyEventV2 } from "aws-lambda";
import { beforeEach, expect, it, vi} from "vitest";
import { DynamoDBClient, GetItemCommand } from '@aws-sdk/client-dynamodb';
import { mockClient } from 'aws-sdk-client-mock';
import { handler } from '../get_trainspace';

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


it("test successful get trainspace call", async () => {
  ddbMock.on(GetItemCommand).resolves({
    Item: { trainspaceID: { S: 'sample trainspace id' } }
  })

  // @ts-expect-error : error doesn't affect functionality. We don't need the rest of the event, and it's really long for no reason
  const event: APIGatewayProxyEventV2 =  {
    pathParameters: {
      id: "some trainspace_id"
    },
  }

  const result = await handler(event);
  expect(result.statusCode).toEqual(200);
});


it("test no existing trainspace id", async () => {
  ddbMock.on(GetItemCommand).resolves({
    Item: undefined
  })

  // @ts-expect-error : error doesn't affect functionality. We don't need the rest of the event, and it's really long for no reason
  const event: APIGatewayProxyEventV2 =  {
    pathParameters: {
      id: "some trainspace_id"
    },
  }
    
  const result = await handler(event);
  expect(result.statusCode).toEqual(404);
});

it("test no trainspace id given", async () => {
  ddbMock.on(GetItemCommand).resolves({
    Item: { trainspaceID: { S: 'sample trainspace id' } }
  })

  // @ts-expect-error : error doesn't affect functionality. We don't need the rest of the event, and it's really long for no reason
  const event: APIGatewayProxyEventV2 =  {
    pathParameters: {
    }
  }
    
  const result = await handler(event);
  expect(result.statusCode).toEqual(401);
});

it("test malformed request", async () => {
  // @ts-expect-error : we are trying to cause an error
  const result = await handler(undefined);
  expect(result.statusCode).toEqual(400);
});