import { APIGatewayProxyEventV2 } from "aws-lambda";
import { beforeEach, expect, it, vi} from "vitest";
import { DynamoDBClient, PutItemCommand } from '@aws-sdk/client-dynamodb';
import { mockClient } from 'aws-sdk-client-mock';
import { handler } from '../create_user';

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

it("test successful create user call", async () => {
  ddbMock.on(PutItemCommand).resolves({
    $metadata: {
      httpStatusCode: 200,
    }
  })
  // @ts-expect-error : doesn't affect functionality. We don't need the rest of the event
  const event: APIGatewayProxyEventV2 =  {
    headers: {
      authorization: 'abcd',
    },
      body: '{\n' +
        '    "name": "SETH SHI BUT UPDATED",\n' +
        '    "email": "TESTEMAIL@GMAIL.COM",\n' +
        '    "phone": "123-456-7890"\n' +
              '}',
  }
  const result = await handler(event);
  expect(result.statusCode).toEqual(200);
});

it("test internal service error", async () => {
    ddbMock.on(PutItemCommand).resolves({
      $metadata: {
        httpStatusCode: 456,
      }
    })
    // @ts-expect-error : error doesn't affect functionality. We don't need the rest of the event
    const event: APIGatewayProxyEventV2 =  {
      headers: {
        authorization: 'abcd',
      },
        body: '{\n' +
          '    "name": "SETH SHI BUT UPDATED",\n' +
          '    "email": "TESTEMAIL@GMAIL.COM",\n' +
          '    "phone": "123-456-7890"\n' +
                '}',
    }
      
    const result = await handler(event);
    expect(result.statusCode).toEqual(500);
  });

it("test undefined event", async () => {
    ddbMock.on(PutItemCommand).resolves({
      $metadata: {
        httpStatusCode: 400,
      }
    })
    // @ts-expect-error : we are trying to cause an error
    const result = await handler(undefined);
    expect(result.statusCode).toEqual(404);
  });