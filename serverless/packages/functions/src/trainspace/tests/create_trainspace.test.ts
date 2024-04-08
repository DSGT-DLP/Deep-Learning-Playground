import { APIGatewayProxyEventV2 } from "aws-lambda";
import { beforeEach, expect, it, vi} from "vitest";
import { PutCommand } from '@aws-sdk/lib-dynamodb';
import { DynamoDBClient} from '@aws-sdk/client-dynamodb';
import { mockClient } from 'aws-sdk-client-mock';
import { handler } from '../create_trainspace';

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

it("test successful create trainspace call", async () => {
  ddbMock.on(PutCommand).resolves({
    $metadata: {
      httpStatusCode: 200,
    }
  })
  //error is fine, doesn't affect functionality. We don't need the rest of the event, and it's really long for no reason
  const event: APIGatewayProxyEventV2 =  {
    headers: {
      authorization: 'abcd',
      
    },
      body: '{\n' +
        '    "name": "SOME NAME",\n' +
        '    "data_source": "SOME DATA SOURCE",\n' +
        '    "dataset_data": "SOME DATA",\n' +
        '    "review_data": "SOME REVIEW DATA",\n' +
        '    "model_id": "SOME MODEL ID",\n' +
        '    "results_s3": "SOME RESULTS"\n' +
              '}',
  }
  const result = await handler(event);
  expect(result.statusCode).toEqual(200);
});

it("test internal service error", async () => {
    ddbMock.on(PutCommand).resolves({
      $metadata: {
        httpStatusCode: 456,
      }
    })
    //error is fine, doesn't affect functionality. We don't need the rest of the event, and it's really long for no reason
    const event: APIGatewayProxyEventV2 =  {
      headers: {
        authorization: 'abcd',
      },
      body: '{\n' +
        '    "name": "SOME NAME",\n' +
        '    "data_source": "SOME DATA SOURCE",\n' +
        '    "dataset_data": "SOME DATA",\n' +
        '    "review_data": "SOME REVIEW DATA",\n' +
        '    "model_id": "SOME MODEL ID",\n' +
        '    "results_s3": "SOME RESULTS"\n' +
              '}',
    }
      
    const result = await handler(event);
    expect(result.statusCode).toEqual(500);
  });

it("test undefined event", async () => {
    ddbMock.on(PutCommand).resolves({
      $metadata: {
        httpStatusCode: 400,
      }
    })
    const result = await handler(undefined);
    expect(result.statusCode).toEqual(404);
  });