import { TrainStatus } from '../trainspace/constants';
import { PutItemCommandInput } from "@aws-sdk/client-dynamodb";

export function create_trainspace(trainspace_id: string, uid: string, data_source: string, dataset_data: object, name: string, parameters_data: object, review_data: string) : PutItemCommandInput | null
{
    let output: PutItemCommandInput = 
    {
        TableName : "trainspace",
        
        Item : 
        {
            trainspace_id : 
            {
                S : trainspace_id
            },
            uid :
            {
                S : uid
            },
            created :
            {
                S : Date.now().toString()
            },
            data_source :
            {
                S : data_source
            },
            dataset_data :
            {
                S : JSON.stringify(dataset_data)
            },
            name :
            {
                S : name
            },
            parameters_data :
            {
                S : JSON.stringify(parameters_data)
            },
            review_data : 
            {
                S : review_data
            },
            status :
            {
                S : TrainStatus.QUEUED
            }
        }
    }
    return output;
}