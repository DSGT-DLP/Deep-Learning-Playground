import { APIGatewayProxyEventV2 } from "aws-lambda";
import parseJwt from "@dlp-sst-app/core/src/parseJwt";
import { v4 as uuidv4 } from "uuid";
import { DynamoDBClient, PutItemCommand } from "@aws-sdk/client-dynamodb";
import { TrainStatus } from "./constants";

export async function handler<APIGatewayProxyHandlerV2>(
  event: APIGatewayProxyEventV2
) {
  if (event) {
    const user_id: string = parseJwt(event.headers.authorization ?? "")[
      "user_id"
    ];
    const eventBody = JSON.parse(event.body ? event.body : "");
    const trainspaceId = uuidv4();

    const client = new DynamoDBClient({});
    const review_data = {
      M: {
        ...("notificationEmail" in eventBody["review_data"] && {
          notification_email: {
            S: eventBody["review_data"]["notificationEmail"],
          },
        }),
        ...("notificationPhoneNumber" in eventBody["review_data"] && {
          notification_phone_number: {
            S: eventBody["review_data"]["notificationPhoneNumber"],
          },
        }),
      },
    };

    const putCommand: PutItemCommand = new PutItemCommand({
      TableName: "TrainspaceTable",
      Item: {
        trainspace_id: { S: trainspaceId },
        name: { S: eventBody["name"] },
        user_id: { S: user_id },
        data_source: { S: eventBody["data_source"] },
        dataset_data: {
          M: {
            name: { S: eventBody["dataset_data"]["name"] },
            is_default_dataset: {
              BOOL: eventBody["dataset_data"]["isDefaultDataset"],
            },
          },
        },
        ...(Object.keys(review_data).length && { review_data }),
        // model_id: { S: eventBody["model_id"] }, # to be implemented
        // results_s3: { S: eventBody["results_s3"] }, # not needed
        status: { S: TrainStatus.QUEUED },
      },
    });
    if (putCommand == null) {
      return {
        statusCode: 400,
        body: JSON.stringify({ message: "Invalid request body" }),
      };
    }

    const response = await client.send(putCommand);

    if (response.$metadata.httpStatusCode != 200) {
      return {
        statusCode: 500,
        body: JSON.stringify({ message: "Internal server error." }),
      };
    }

    return {
      statusCode: 200,
      body: JSON.stringify({
        trainspaceId: trainspaceId,
        message: "Successfully created a new trainspace.",
      }),
    };
  }
  return {
    statusCode: 404,
    body: JSON.stringify({ message: "Event not Found" }),
  };
}
function removeUndefinedValues(obj: { [key: string]: any }) {
  const newObj: { [key: string]: any } = {};
  for (const key in obj) {
    if (obj[key] !== undefined) {
      newObj[key] = obj[key];
    }
  }
  return newObj;
}
