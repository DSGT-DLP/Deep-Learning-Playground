import { APIGatewayProxyEventV2 } from "aws-lambda";
import { beforeEach, expect, it, vi} from "vitest";
import { DynamoDBClient, QueryCommand, BatchExecuteStatementCommand } from '@aws-sdk/client-dynamodb';
import { mockClient } from 'aws-sdk-client-mock';
import { handler } from '../delete_all_trainspace';

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
  ddbMock.on(QueryCommand).resolves({
    "Items": [{
      "trainspace_id": { "S": "test id" },
    }],
    "Count": 1
  });
  ddbMock.on(BatchExecuteStatementCommand).resolves({
    $metadata: {
      httpStatusCode: 200,
    }
  })

  // @ts-expect-error
  const event: APIGatewayProxyEventV2 =  {
    headers: {
      authorization: 'abcd',
    },
  }
  const result = await handler(event);
  expect(result.statusCode).toEqual(200);
});


it("test no batch delete response call", async () => {
  ddbMock.on(QueryCommand).resolves({
    "Items": [{
      "user_id":{"S":"abcd"},
      "trainspace_id": { "S": "test id" },
    }],
    "Count": 4
  });
    ddbMock.on(BatchExecuteStatementCommand).resolves({
      $metadata: {
        httpStatusCode: undefined,
      }
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

it("test incorrect batch delete response failed call", async () => {
  ddbMock.on(QueryCommand).resolves({
    "Items": [{
      "user_id":{"S":"abcd"},
      "trainspace_id": { "S": "test id" },
    }],
    "Count": 4
  });
    ddbMock.on(BatchExecuteStatementCommand).resolves({
      $metadata: {
        httpStatusCode: 267,
      }
    })

    // @ts-expect-error
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
  ddbMock.on(BatchExecuteStatementCommand).resolves({
    $metadata: {
      httpStatusCode: 200,
    }
  })

  //@ts-expect-error
  const event: APIGatewayProxyEventV2 =  {
    headers: {
      authorization: 'abcd',
    },
  }
    
  const result = await handler(event);
  expect(result.statusCode).toEqual(405);
});

it("test malformed call", async () => {

  // @ts-expect-error : we are trying to cause an error
    const result = await handler(undefined);
    expect(result.statusCode).toEqual(400);
});