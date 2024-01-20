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
        let putCommandInput: PutCommandInput = {
            TableName: "trainspace",
            Item:
            {
                //list of models
                //list of blocks
                trainspace_id: trainspaceId,
                created: Date.now().toString(),
                uid: user_id,
                //Source (default, s3)
                //dataset id
                //s3 url (if s3 source)
                //should this be a list?
                dataset_data: eventBody['dataset_source'] == 's3' ? {
                    data_source: eventBody['dataset_source'],
                    dataset_id: eventBody['dataset_id'],
                    s3_url: eventBody['dataset_source'].s3_url,
                } : {
                    data_source: eventBody['dataset_source'],
                    dataset_id: eventBody['dataset_id'],
                },
                parameters_data: eventBody['data_source'] == 'IMAGE' ? JSON.stringify(eventBody['parameters_data']) : { 
                    criterion: eventBody['criterion'],
                    optimizer_name: eventBody['optimizer_name'],
                    shuffle: eventBody['shuffle'],
                    epochs: eventBody['epochs'],
                    batch_size: eventBody['batch_size'],
                    user_arch: eventBody['user_arch']
                },
                /**
                 * Name
                 * Model id
                 * Dataset id
                 * list of model versions:
                 * Model versions:
                 *      Name of model version
                 *      s3 url to mode
                 */
                models: {
                    model_name: eventBody['model_name'],
                    model_id: eventBody['model_id'],
                    dataset_id: eventBody['dataset_id'],
                    model_versions: {
                        model_name: eventBody['model_name'],
                        s3_url: eventBody['model_s3_url']
                    }
                },
                /**
                 * Blocks (stored within trainspace item):
                Block id
                Type of block
                Other info based on the type of block
                Run data
                    s3 url, string, number, etc based on the type of block
                 */
                blocks: {
                    block_id: eventBody['block_id'],
                    block_type: eventBody['block_type'],
                    //not sure what to do with "other info based on the type of block"
                    run_data: {
                        s3_url: eventBody['run_s3_url'],
                        string: eventBody['run_string'],
                        number: eventBody['run_number']
                    }
                },
                //keep?
                name: eventBody['name'],
                review_data: eventBody['review_data'],
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