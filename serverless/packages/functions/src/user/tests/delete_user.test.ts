import { APIGatewayProxyEventV2 } from "aws-lambda";
import { beforeEach, expect, it, vi} from "vitest";
import { DynamoDBClient, DeleteItemCommand } from '@aws-sdk/client-dynamodb';
import { mockClient } from 'aws-sdk-client-mock';
import { handler } from '../delete_user';

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

it("test successful delete user call", async () => {
  ddbMock.on(DeleteItemCommand).resolves({
    $metadata: {
      httpStatusCode: 200,
    }
  })
  // @ts-expect-error : error doesn't affect functionality. We don't need the rest of the event, and it's really long for no reason
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

it("test no response failed operation call", async () => {
    ddbMock.on(DeleteItemCommand).resolves({
      $metadata: {
        httpStatusCode: undefined,
      }
    })
    
    // @ts-expect-error : error doesn't affect functionality. We don't need the rest of the event, and it's really long for no reason
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
    expect(result.statusCode).toEqual(404);
});

it("test different status code failed operation call", async () => {
    ddbMock.on(DeleteItemCommand).resolves({
      $metadata: {
        httpStatusCode: 267,
      }
    })
    // @ts-expect-error : error doesn't affect functionality. We don't need the rest of the event, and it's really long for no reason
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
    expect(result.statusCode).toEqual(404);
});

it("test malformed call", async () => {
  // @ts-expect-error : we are trying to cause an error
  const result = await handler(undefined);
  expect(result.statusCode).toEqual(400);
});