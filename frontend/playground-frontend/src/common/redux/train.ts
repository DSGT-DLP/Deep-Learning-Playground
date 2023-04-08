import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import * as XLSX from "xlsx";

type InitState = {
  customModelName: string;
  fileName?: string;
  csvDataInput: object[];
  oldCsvDataInput: object[];
  uploadedColumns: object[];
  fileURL?: string;
};

const initialState: InitState = {
  customModelName: `Model ${new Date().toLocaleString()}`,
  csvDataInput: [],
  oldCsvDataInput: [],
  uploadedColumns: [],
};

export const trainSlice = createSlice({
  name: "train",
  initialState: initialState,
  reducers: {
    setModelName: (state, action: PayloadAction<string>) => {
      console.log(action.payload);
      state.customModelName = action.payload;
    },
    handleFileUpload: (state, action: PayloadAction<File>) => {
      const csvToJson = (dataString: string) => {
        const dataStringLines = dataString.split(/\r\n|\n/);
        const headers = dataStringLines[0].split(
          /,(?![^"]*"(?:(?:[^"]*"){2})*[^"]*$)/
        );

        const list = [];
        for (let i = 1; i < dataStringLines.length; i++) {
          const row = dataStringLines[i].split(
            /,(?![^"]*"(?:(?:[^"]*"){2})*[^"]*$)/
          );
          if (headers && row.length === headers.length) {
            const obj: { [x: string]: unknown } = {};
            for (let j = 0; j < headers.length; j++) {
              let d = row[j];
              if (d.length > 0) {
                if (d[0] === '"') d = d.substring(1, d.length - 1);
                if (d[d.length - 1] === '"') d = d.substring(d.length - 2, 1);
              }
              if (headers[j]) {
                obj[headers[j]] = d;
              }
            }

            // remove the blank rows
            if (Object.values(obj).filter((x) => x).length > 0) {
              list.push(obj);
            }
          }
        }

        // prepare columns list from headers
        const columns = headers.map((c) => ({
          name: c,
          selector: (row: { [x: string]: unknown }) => row[c],
        }));
        return [list, columns];
      };

      const file = action.payload;
      state.fileName =
        file.name == null
          ? undefined
          : file.name.replace(/\.[^/.]+$/, "").substring(0, 20) +
            "." +
            file.name.split(".").pop();
      const reader = new FileReader();
      reader.onload = async (evt) => {
        /* Parse data */
        const bstr = evt.target?.result;
        const wb = XLSX.read(bstr, { type: "binary" });
        /* Get first worksheet */
        const wsname = wb.SheetNames[0];
        const ws = wb.Sheets[wsname];
        /* Convert array of arrays */
        const data = XLSX.utils.sheet_to_csv(ws);
        const [list, columns] = csvToJson(data);
        state.csvDataInput = list;
        state.uploadedColumns = columns;
        state.oldCsvDataInput = list;
      };
    },
    resetTrain: (state) => {
      console.log("new");
      state.customModelName = `Model ${new Date().toLocaleString()}`;
      console.log(state.customModelName);
    },
  },
});

export const { setModelName, resetTrain, handleFileUpload } =
  trainSlice.actions;

export default trainSlice.reducer;
