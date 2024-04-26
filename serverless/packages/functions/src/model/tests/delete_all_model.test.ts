import { APIGatewayProxyEventV2 } from "aws-lambda";
import { beforeEach, expect, it, vi} from "vitest";
import { DynamoDBClient, QueryCommand, BatchExecuteStatementCommand } from '@aws-sdk/client-dynamodb';
import { mockClient } from 'aws-sdk-client-mock';
import { handler } from '../delete_all_model';


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

it("test successful delete all model call", async () => {
  ddbMock.on(QueryCommand).resolves({
    "Items": [{
      "model_id": { "S": "test id" },
    }],
    "Count": 1
  });
  ddbMock.on(BatchExecuteStatementCommand).resolves({
    $metadata: {
      httpStatusCode: 200,
    }
  })

  // @ts-expect-error : error doesn't affect functionality. We don't need the rest of the event, and it's really long for no reason
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
      "model_id": { "S": "test id" },
    }],
    "Count": 4
  });
    ddbMock.on(BatchExecuteStatementCommand).resolves({
      $metadata: {
        httpStatusCode: undefined,
      }
    })
    
    // @ts-expect-error : error doesn't affect functionality. We don't need the rest of the event, and it's really long for no reason
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
      "model_id": { "S": "test id" },
    }],
    "Count": 4
  });
    ddbMock.on(BatchExecuteStatementCommand).resolves({
      $metadata: {
        httpStatusCode: 267,
      }
    })

    // @ts-expect-error : error doesn't affect functionality. We don't need the rest of the event, and it's really long for no reason
    const event: APIGatewayProxyEventV2 =  {
      headers: {
        authorization: 'abcd',
      },
    }
      
    const result = await handler(event);
    expect(result.statusCode).toEqual(404);
});


it("test delete all on no existing model call", async () => {
  ddbMock.on(QueryCommand).resolves({
    "Items": [],
    "Count": 0
  });
  ddbMock.on(BatchExecuteStatementCommand).resolves({
    $metadata: {
      httpStatusCode: 200,
    }
  })

  // @ts-expect-error : error doesn't affect functionality. We don't need the rest of the event, and it's really long for no reason
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