import React from "react";
import PropTypes from "prop-types";
import * as XLSX from "xlsx";
import { FaCloudUploadAlt } from "react-icons/fa";
//import { getSignedUploadUrl } from "../helper_functions/TalkWithBackend";
//import axios from "axios";

// src: https://www.cluemediator.com/read-csv-file-in-react
const CSVInputFile = (props) => {
  const { setData, setColumns, setOldData, fileName, setFileName } = props;

  // process CSV data
  const csvToJson = (dataString) => {
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
        const obj = {};
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
      selector: (row) => row[c],
    }));
    return [list, columns];
  };

  // handle file upload
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    setFileName(
      file.name == null
        ? null
        : file.name.replace(/\.[^/.]+$/, "").substring(0, 20) +
            "." +
            file.name.split(".").pop()
    );
    const reader = new FileReader();
    reader.onload = async (evt) => {
      /* Parse data */
      const bstr = evt.target.result;
      const wb = XLSX.read(bstr, { type: "binary" });
      /* Get first worksheet */
      const wsname = wb.SheetNames[0];
      const ws = wb.Sheets[wsname];
      /* Convert array of arrays */
      const data = XLSX.utils.sheet_to_csv(ws);
      const [list, columns] = csvToJson(data);
      setData(list);
      setColumns(columns);
      setOldData(list);
    };
    reader.readAsBinaryString(file);
    /*
    const reader2 = new FileReader();
    reader2.readAsDataURL(file);
    reader2.onload = async (evt) => {
      const buffer = evt.target.result;
      console.log(buffer);
    };*/
    /* let form = new FormData();
    form.append("file", file);
    const response = await getSignedUploadUrl(1, file.name, file);
    console.log(response);
    const url = new URL(response["url"] + response["fields"]["key"]);
    url.searchParams.append(
      "AWSAccessKeyId",
      response["fields"]["AWSAccessKeyId"]
    );
    url.searchParams.append("policy", response["fields"]["policy"]);
    url.searchParams.append("signature", response["fields"]["signature"]);

    console.log(
      await fetch(url, {
        method: "PUT",
        //headers: new Headers({ "Content-Type": "application/json" }),
        body: form,
      })
    ); */
    /*
    const response = await fetch(`${endpoint}`);
    //?version=1&filename=${new URLSearchParams({version: 1}).toString()}
    const resJson = await response.json();
    console.log(response);
    console.log(resJson);*/
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
        onChange={handleFileUpload}
        style={{ width: "100%" }}
      />
    </>
  );
};

CSVInputFile.propTypes = {
  setData: PropTypes.func.isRequired,
  setColumns: PropTypes.func.isRequired,
  setOldData: PropTypes.func.isRequired,
  fileName: PropTypes.any,
  setFileName: PropTypes.func.isRequired,
};

export default CSVInputFile;
