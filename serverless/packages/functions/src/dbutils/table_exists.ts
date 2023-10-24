import { get_all_tablenames } from "./get_all_tablenames";

export async function table_exists(table_name: string) : Promise<boolean> {
    const tableNames: string[] = await get_all_tablenames();
    return tableNames.findIndex(storedTableName => storedTableName.endsWith(table_name)) !== -1;
}
