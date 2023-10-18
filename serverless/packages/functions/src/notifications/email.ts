import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import {
  SESv2Client,
  SendEmailCommand,
  SendEmailCommandInput,
} from "@aws-sdk/client-sesv2";
import assert from "assert";

const DLP_EMAIL = "dlp@datasciencegt.org";

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
  const canSendEmail = event && event.body;

  if (!canSendEmail) {
    return {
      statusCode: 404,
      body: JSON.stringify({ message: "Not Found" }),
    };
  }

  assert(event.body, "event.body is undefined");
  const bodyJson = JSON.parse(event.body);

  const {
    subject,
    bodyHtml,
    bodyRawText,
    attachmentArray,
    toAddresses, // comma separated list of email addresses e.g. a@gmail.com,b.gmail.com
    ccAddresses,
    bccAddresses,
  } = bodyJson;

  const client = new SESv2Client();
  const input: SendEmailCommandInput = {
    FromEmailAddress: DLP_EMAIL,
    Destination: {
      ToAddresses: toAddresses,
      CcAddresses: ccAddresses,
      BccAddresses: bccAddresses,
    },
    Content: {
      Simple: {
        Subject: { Data: subject },
        Body: {
          Text: { Data: bodyRawText || bodyHtml || "", Charset: "UTF-8" },
          Html: { Data: bodyHtml || bodyRawText || "", Charset: "UTF-8" },
        },
      },
    },
    ReplyToAddresses: [DLP_EMAIL],
  };

  const command = new SendEmailCommand(input);
  const response = await client.send(command);

  const success = response.$metadata.httpStatusCode === 200;

  if (!success) {
    return {
      statusCode: 500,
      body: JSON.stringify({ message: "Failed", errorMessage: response }),
    };
  }

  return {
    statusCode: 200,
    body: JSON.stringify({ message: "Success", data: response }),
  };
};
