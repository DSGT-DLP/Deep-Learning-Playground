import React from "react";
import PropTypes from "prop-types";
import { toast } from "react-toastify";

const HomeCSVInputURL = (props) => {
  const { fileURL, setFileURL, setCSVColumns, setCSVDataInput } = props;

  async function handleURL(url) {
    try {
      if (!url) return;

      let response = await fetch(url);
      response = await response.text();
      const responseLines = response.split(/\r\n|\n/);
      const headers = responseLines[0].split(
        /,(?![^"]*"(?:(?:[^"]*"){2})*[^"]*$)/
      );

      const list = [];
      for (let i = 1; i < responseLines.length; i++) {
        const row = responseLines[i].split(
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

        // prepare columns list from headers
        const columns = headers.map((c) => ({
          name: c,
          selector: (row) => row[c],
        }));

        setCSVDataInput(list);
        setCSVColumns(columns);
        setFileURL(url);
      }
    } catch (e) {
      toast.error("Incorrect URL");
    }
  }
  return (
    <input
      style={{ width: "100%" }}
      placeholder="Or type in URL..."
      value={fileURL}
      onChange={(e) => setFileURL(e.target.value)}
      onBlur={(e) => handleURL(e.target.value)}
    />
  );
};

HomeCSVInputURL.propTypes = {
  fileURL: PropTypes.string.isRequired,
  setFileURL: PropTypes.func.isRequired,
  setCSVColumns: PropTypes.func.isRequired,
  setCSVDataInput: PropTypes.func.isRequired,
};

export default HomeCSVInputURL;
