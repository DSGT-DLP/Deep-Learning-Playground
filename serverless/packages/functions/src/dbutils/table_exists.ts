import { get_all_tablenames } from "./get_all_tablenames";
import { DynamoDBClient, DescribeTableCommand } from '@aws-sdk/client-dynamodb';

export async function table_exists(table_name: string) : Promise<boolean> {
    const tableNames: string[] = await get_all_tablenames();
    return tableNames.findIndex(storedTableName => storedTableName.endsWith(table_name)) !== -1;
}

export async function table_attributes_exists(table_name: string, table_attributes: object) : Promise<boolean>
{
    const client = new DynamoDBClient();
    const describeTableCommand = new DescribeTableCommand({
        "TableName": table_name
    });

}