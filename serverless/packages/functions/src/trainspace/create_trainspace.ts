import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import parseJwt from "@dlp-sst-app/core/parseJwt";
import { v4 as uuidv4 } from 'uuid';
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, PutCommand, PutCommandInput } from '@aws-sdk/lib-dynamodb';
import { TrainStatus } from './constants';

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
    if (event) {
        const user_id: string = parseJwt(event.headers.authorization ?? "")["user_id"];
        const eventBody = JSON.parse(event.body? event.body : "");
        const trainspaceId = uuidv4();

        const datasetArray = eventBody['datasets'];
        const modelArray = eventBody['models'];
        const blockArray = eventBody['blocks'];
        
        let putCommandInput: PutCommandInput = {
            TableName: "TrainspaceTable",
            Item:
            {
                trainspace_id: trainspaceId,
                created: Date.now().toString(),
                uid: user_id,

                datasets: datasetArray.map((eventBody: {[x: string] : any;}) => ({
                    dataset: removeUndefinedValues(eventBody['dataset_source'] == 's3' ? {
                        data_source: eventBody['dataset_source']?.trim(),
                        dataset_id: eventBody['dataset_id']?.trim(),
                        s3_url: eventBody['s3_url']?.trim(),
                    } : {
                        data_source: eventBody['dataset_source']?.trim(),
                        dataset_id: eventBody['dataset_id']?.trim()
                    })
                })),
                models: modelArray.map((eventBody: { [x: string]: any; }) => ({
                    model: removeUndefinedValues({
                      model_name: eventBody['model_name']?.trim(),
                      model_id: eventBody['model_id']?.trim(),
                      //dataset_id: eventBody['dataset_id']?.trim(),
                      model_versions: {
                        model_name: eventBody['model_name']?.trim(),
                        s3_url: eventBody['model_s3_url']?.trim()
                      }
                    })
                  })),
                blocks: blockArray.map((eventBody: {[x: string] : any;}) => ({
                    block: removeUndefinedValues({
                        block_id: eventBody['block_id']?.trim(),
                        block_type: eventBody['block_type']?.trim(),
                        //not sure what to do with "other info based on the type of block"
                        run_data: {
                            s3_url: eventBody['block_s3_url']?.trim(),
                            string: eventBody['run_string']?.trim(),
                            number: eventBody['run_number']?.trim()
                        }
                    })
                })),
                status: TrainStatus.QUEUED
            }
        }

        if (putCommandInput == null)
        {
            return {
                statusCode: 400,
                body: JSON.stringify({ message: "Invalid request body" })
            }
        }

        const client = new DynamoDBClient({});
        const docClient = DynamoDBDocumentClient.from(client);

        const command = new PutCommand(putCommandInput);
        const response = await docClient.send(command);

        if (response.$metadata.httpStatusCode != 200) {
            return {
                statusCode: 500,
                body: JSON.stringify({ message: "Internal server error."})
            };
        }
        
        return {
            statusCode: 200,
            body: JSON.stringify({ trainspaceId: trainspaceId, message: "Successfully created a new trainspace."})
        };
      }
    return {
        statusCode: 404,
        body: JSON.stringify({ message: "Not Found" }),
    };
};
function removeUndefinedValues(obj: { [key: string]: any }) {
    const newObj: { [key: string]: any } = {};
    for (const key in obj) {
        if (obj[key] !== undefined) {
            newObj[key] = obj[key];
        }
    }
    return newObj;
}