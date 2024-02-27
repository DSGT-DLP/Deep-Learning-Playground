import { APIGatewayProxyHandlerV2, APIGatewayProxyEventV2 } from "aws-lambda";
import { beforeEach, expect, it, vi} from "vitest";
import {DynamoDBClient} from '@aws-sdk/client-dynamodb';
import {DeleteCommand, DynamoDBDocumentClient, GetCommand, PutCommand } from '@aws-sdk/lib-dynamodb';
import parseJwt from "../../../../core/src/parseJwt";
import {mockClient} from 'aws-sdk-client-mock';
import { handler } from '../delete_user';
import { NativeAttributeValue } from "@aws-sdk/util-dynamodb";
import 'aws-sdk-client-mock-vitest';



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

it("test successful delete user call", async () => {
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
    ddbMock.on(DeleteCommand).resolves({
      $metadata: {
        httpStatusCode: undefined,
      }
    })
    
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
    const result = await handler(undefined);
    expect(result.statusCode).toEqual(400);
});