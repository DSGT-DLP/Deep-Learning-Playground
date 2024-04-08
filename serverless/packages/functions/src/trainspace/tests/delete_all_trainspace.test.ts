import { APIGatewayProxyEventV2 } from "aws-lambda";
import { beforeEach, expect, it, vi} from "vitest";
import { QueryCommand, DeleteCommand } from '@aws-sdk/lib-dynamodb';
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { mockClient } from 'aws-sdk-client-mock';
import { handler } from '../delete_all';


//mocks parseJwt so that the call just returns whatever the input is
vi.mock('@dlp-sst-app/core/src/parseJwt', async () => {
  return {
    default: (input: String) => ({ user_id: input })
  }
});

beforeEach(async () => {
  ddbMock.reset();
});

const ddbMock = mockClient(DynamoDBClient);

it("test successful delete all trainspace call", async () => {
  console.log("at least we're running");
  ddbMock.on(QueryCommand).resolves({
    "Items": [{
      "trainspace_id": { "S": "test id" },
    }],
    "Count": 1
  });
  ddbMock.on(DeleteCommand).resolves({
    $metadata: {
      httpStatusCode: 201,
    }
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


it("test no response failed operation call", async () => {
  ddbMock.on(QueryCommand).resolves({
    "Items": [{
      "user_id":{"S":"abcd"},
      "trainspace_id": { "S": "test id" },
    }],
    "Count": 4
  });
    ddbMock.on(DeleteCommand).resolves({
      $metadata: {
        httpStatusCode: undefined,
      }
    })
    
    const event: APIGatewayProxyEventV2 =  {
      headers: {
        authorization: 'abcd',
      },
    }

    const result = await handler(event);
    expect(result.statusCode).toEqual(404);
});


it("test delete all on no existing trainspaces call", async () => {
  ddbMock.on(QueryCommand).resolves({
    "Items": [],
    "Count": 0
  });
  ddbMock.on(DeleteCommand).resolves({
    $metadata: {
      httpStatusCode: 200,
    }
  })
  //error is fine, doesn't affect functionality. We don't need the rest of the event, and it's really long for no reason
  const event: APIGatewayProxyEventV2 =  {
    headers: {
      authorization: 'abcd',
    },
  }
    
  const result = await handler(event);
  expect(result.statusCode).toEqual(405);
});


it("test different status code failed operation call", async () => {
  ddbMock.on(QueryCommand).resolves({
    "Items": [{
      "user_id":{"S":"abcd"},
      "trainspace_id": { "S": "test id" },
    }],
    "Count": 4
  });
    ddbMock.on(DeleteCommand).resolves({
      $metadata: {
        httpStatusCode: 267,
      }
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


it("test malformed call", async () => {
    const result = await handler(undefined);
    expect(result.statusCode).toEqual(400);
});