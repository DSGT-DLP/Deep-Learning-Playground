import { APIGatewayProxyEventV2 } from "aws-lambda";
import { S3Client, GetObjectCommand } from "@aws-sdk/client-s3";
import { DynamoDBClient, GetItemCommand } from "@aws-sdk/client-dynamodb";

export async function handler<APIGatewayProxyHandlerV2>(
  event: APIGatewayProxyEventV2
) {
  let pathParams = null;
  if (event && (pathParams = event["pathParameters"]) != null) {
    const trainspaceId: string | undefined = pathParams["id"];
    const queryParams = event["queryStringParameters"];
    let withResults = false;
    if (queryParams !== undefined) {
      withResults = "with_results" in queryParams ? queryParams["with_results"] === 'true' : false;
    }

    if (trainspaceId === undefined) {
      return {
        statusCode: 401,
        body: JSON.stringify({
          message: "Malformed request content - trainspace ID missing.",
        }),
      };
    }

    const client: DynamoDBClient = new DynamoDBClient({});

    const command: GetItemCommand = new GetItemCommand({
      TableName: "TrainspaceTable",
      Key: {
        trainspace_id: { S: trainspaceId },
      },
    });

    const response = await client.send(command);
    if (!response.Item) {
      return {
        statusCode: 404,
        body: JSON.stringify({
          message: "Provided trainspaceId does not exist",
        }),
      };
    }

    let detailedTrainResultsData;
    if (withResults) {
      const client = new S3Client();

      const res = await client.send(
        new GetObjectCommand({
          Bucket: "dlp-executions",
          Key: `${trainspaceId}.json`,
        })
      );
      const bodyString = await res.Body.transformToString();
      detailedTrainResultsData = JSON.parse(bodyString);
    }

    return {
      statusCode: 200,
      body: JSON.stringify({
        message: "Successfully retrieved trainspace data",
        trainspace: {
          config: response.Item,
          detailedTrainResultsData: detailedTrainResultsData,
        },
      }),
    };
  }
  return {
    statusCode: 400,
    body: JSON.stringify({ message: "Malformed request content" }),
  };
}
