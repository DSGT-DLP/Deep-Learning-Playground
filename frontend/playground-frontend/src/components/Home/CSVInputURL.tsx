import React from "react";
import { toast } from "react-toastify";
import { Form } from "react-bootstrap";
import { CSVInputDataColumnType, CSVInputDataRowType } from "./CSVInputFile";

interface CSVInputURLPropTypes {
  fileURL: string;
  setFileURL: React.Dispatch<React.SetStateAction<string>>;
  setCSVColumns: React.Dispatch<React.SetStateAction<CSVInputDataColumnType[]>>;
  setCSVDataInput: React.Dispatch<React.SetStateAction<CSVInputDataRowType[]>>;
  setOldCSVDataInput: React.Dispatch<
    React.SetStateAction<CSVInputDataRowType[]>
  >;
}
const HomeCSVInputURL = (props: CSVInputURLPropTypes) => {
  const {
    fileURL,
    setFileURL,
    setCSVColumns,
    setCSVDataInput,
    setOldCSVDataInput,
  } = props;

  async function handleURL(url: string) {
    try {
      if (!url) return;

      const response = await fetch(url);
      const responseText = await response.text();
      const responseLines = responseText.split(/\r\n|\n/);
      const headers = responseLines[0].split(
        /,(?![^"]*"(?:(?:[^"]*"){2})*[^"]*$)/
      );

      const list = [];
      for (let i = 1; i < responseLines.length; i++) {
        const row = responseLines[i].split(
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

        // prepare columns list from headers
        const columns = headers.map((c) => ({
          name: c,
          selector: (row: CSVInputDataRowType) => row[c],
        }));

        setCSVDataInput(list);
        setCSVColumns(columns);
        setOldCSVDataInput(list);
        setFileURL(url);
      }
    } catch (e) {
      toast.error("Incorrect URL");
    }
  }
  return (
    <Form.Control
      placeholder="Or type in URL..."
      size="sm"
      value={fileURL}
      onChange={(e) => setFileURL(e.target.value)}
      onBlur={(e) => handleURL(e.target.value)}
    />
  );
};

export default HomeCSVInputURL;
