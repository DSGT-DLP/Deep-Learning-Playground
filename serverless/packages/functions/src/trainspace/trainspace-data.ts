import { TrainStatus } from '../constants';
import { PutItemCommandInput } from "@aws-sdk/client-dynamodb";

export default class TrainspaceData
{
    trainspace_id: string;
    uid: string;
    created: string = "";
    data_source: string = "";
    dataset_data: object = {};
    name: string = "";
    parameters_data: object = {};
    review_data: string = "";
    status: TrainStatus = TrainStatus.QUEUED;

    constructor(trainspace_id: string, uid: string) {
        this.trainspace_id = trainspace_id;
        this.uid = uid;
    }

    convertToDynamoItemInput(tableName: string) : PutItemCommandInput {
        let output: PutItemCommandInput = 
        {
            TableName : tableName,
            
            Item : 
            {
                trainspace_id : 
                {
                    S : this.trainspace_id
                },
                uid :
                {
                    S : this.uid
                },
                created :
                {
                    S : this.created
                },
                data_source :
                {
                    S : this.data_source
                },
                dataset_data :
                {
                    S : JSON.stringify(this.dataset_data)
                },
                name :
                {
                    S : this.name
                },
                parameters_data :
                {
                    S : JSON.stringify(this.parameters_data)
                },
                review_data : 
                {
                    S : this.review_data
                },
                status :
                {
                    S : this.status.toString()
                }
            }
        }
        return output;
    }

}