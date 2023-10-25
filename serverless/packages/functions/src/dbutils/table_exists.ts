import { get_all_tablenames } from "./get_all_tablenames";
import { DynamoDBClient, DescribeTableCommand } from '@aws-sdk/client-dynamodb';

export async function table_exists(table_name: string) : Promise<[boolean, string]> {
    const tableNames: string[] = await get_all_tablenames();
    const index: number = tableNames.findIndex(storedTableName => storedTableName.endsWith(table_name));
    return [index !== -1, index !== -1 ? tableNames[index] : ""];
}

export async function table_attributes_exists(tableExistenceInfo: [boolean, string], table_attributes: object) : Promise<boolean>
{
    if (!tableExistenceInfo[0])
    {
        /* If table doesn't exist, return false. */
        return false;
    }
    const client = new DynamoDBClient({});
    const describeTableCommand = new DescribeTableCommand({
        "TableName": tableExistenceInfo[1]
    });

    const response = await client.send(describeTableCommand);
    return true;
}