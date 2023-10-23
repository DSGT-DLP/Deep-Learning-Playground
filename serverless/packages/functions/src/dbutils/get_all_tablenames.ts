
export async function get_all_tablenames() : Promise<string[]> {
    const AWS = require("aws-sdk");
    AWS.config.update({region:'us-east-1'});
    let dynamodb = new AWS.DynamoDB();

    let params = 
    {
        ExclusiveStartTableName: null
    };
    let tables: string[] = [];
    while (true) {
        let response = await dynamodb.listTables(params).promise();
        tables = tables.concat(response.TableNames);

        if (response.LastEvaluatedTableName === undefined) {
            break;
        }
        params.ExclusiveStartTableName = response.LastEvaluatedTableName;
    }
    return tables;
}