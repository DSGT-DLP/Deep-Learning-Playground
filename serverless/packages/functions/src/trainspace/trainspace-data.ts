import { TrainStatus } from '../constants';

export default class TrainspaceData
{
    trainspace_id: string;
    uid: string;
    created: string = "";
    data_source: string = "";
    dataset_data: object = null;
    name: string = "";
    parameters_data: object = null;
    review_data: string = "";
    status: TrainStatus = TrainStatus.QUEUED;

    constructor(trainspace_id: string, uid: string) {
        this.trainspace_id = trainspace_id;
        this.uid = uid;
    }

}