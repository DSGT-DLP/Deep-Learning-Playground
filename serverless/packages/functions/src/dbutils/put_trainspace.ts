import { TrainStatus } from '../trainspace/constants';
import { PutCommandInput } from "@aws-sdk/lib-dynamodb";

export function create_trainspace(trainspaceId: string, user_id: string, dataSource: string, dataset_data: object, trainspaceName: string, parameters_data: object, reviewData: string) : PutCommandInput | null
{
    let output: PutCommandInput = 
    {
        TableName : "trainspace",
        
        Item : 
        {
            trainspace_id : trainspaceId,
            uid : user_id,
            created : Date.now().toString(),
            data_source : dataSource,
            dataset_data : JSON.stringify(dataset_data),
            name : trainspaceName,
            parameters_data : JSON.stringify(parameters_data),
            review_data : reviewData,
            status : TrainStatus.QUEUED
        }
    }
    return output;
}