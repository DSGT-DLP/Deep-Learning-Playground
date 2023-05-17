import React from "react";
import PropTypes from "prop-types";
import * as XLSX from "xlsx";
import { FaCloudUploadAlt } from "react-icons/fa";

interface CSVInputFilePropTypes {
  setData: React.Dispatch<React.SetStateAction<CSVInputDataRowType[]>>;
  setColumns: React.Dispatch<React.SetStateAction<CSVInputDataColumnType[]>>;
  setOldData: React.Dispatch<React.SetStateAction<CSVInputDataRowType[]>>;
  fileName: string;
  setFileName: React.Dispatch<React.SetStateAction<string>>;
}

export interface CSVInputDataRowType {
  [header: string]: string;
}
export interface CSVInputDataColumnType {
  name: string;
  selector: (row: CSVInputDataRowType) => string;
}

const CSVInputFile = (props: CSVInputFilePropTypes) => {
  const { setData, setColumns, setOldData, fileName, setFileName } = props;

  // process CSV data
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
        const obj: CSVInputDataRowType = {};
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
      selector: (row: Record<string, string>) => row[c],
    }));
    return [list, columns];
  };

  // handle file upload
  const handleFileUpload = (files: FileList | null) => {
    if (files) {
      const file = files[0];
      setFileName(
        file.name.replace(/\.[^/.]+$/, "").substring(0, 20) +
          "." +
          file.name.split(".").pop()
      );
      const reader = new FileReader();
      reader.onload = async (evt) => {
        /* Parse data */
        if (evt.target) {
          const bstr = evt.target.result;
          const wb = XLSX.read(bstr, { type: "binary" });
          /* Get first worksheet */
          const wsname = wb.SheetNames[0];
          const ws = wb.Sheets[wsname];
          /* Convert array of arrays */
          const data = XLSX.utils.sheet_to_csv(ws);
          const [list, columns] = csvToJson(data);
          setData(list as CSVInputDataRowType[]);
          setColumns(columns as CSVInputDataColumnType[]);
          setOldData(list as CSVInputDataRowType[]);
        }
      };
      reader.readAsBinaryString(file);
    }
  };

  return (
    <>
      <label
        htmlFor="csv-upload"
        className="custom-file-upload d-flex align-items-center"
      >
        <FaCloudUploadAlt className="me-2" />
        {fileName || "Upload CSV"}
      </label>
      <input
        type="file"
        id="csv-upload"
        accept=".csv,.xlsx,.xls"
        onChange={(e) => handleFileUpload(e.target.files)}
        style={{ width: "100%" }}
      />
    </>
  );
};

CSVInputFile.propTypes = {
  setData: PropTypes.func.isRequired,
  setColumns: PropTypes.func.isRequired,
  setOldData: PropTypes.func.isRequired,
  fileName: PropTypes.string,
  setFileName: PropTypes.func.isRequired,
};

export default CSVInputFile;
