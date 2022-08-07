import React, { useEffect, useState } from "react";
import { PropTypes } from "prop-types";
import { FaCloudUploadAlt } from "react-icons/fa";

const { Dropzone } = require("dropzone");

const LargeFileUpload = (props) => {
  const { setDataUploaded } = props;
  const [fileName, setFileName] = useState();

  const script = document.createElement("script");

  script.src =
    "https://cdnjs.cloudflare.com/ajax/libs/dropzone/5.9.3/dropzone-amd-module.min.js";
  script.async = true;

  document.body.appendChild(script);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    setFileName(file.name.substring(0, 20));
  };

  useEffect(() => {
    Dropzone.autoDiscover = false;
    Dropzone.options.dropper = {
      paramName: "file",
      chunking: true,
      forceChunking: true,
      url: "/upload",
      maxFilesize: 5000, // megabytes
      chunkSize: 1000000, // bytes
      init: function () {
        2 + 2;
      },
    };
    Dropzone.options.dropper.init();
  });

  return (
    <>
      <iframe
        name="dummyframe"
        id="dummyframe"
        style={{ display: "none" }}
      ></iframe>
      <form
        action="/upload"
        encType="multipart/form-data"
        method="POST"
        target="dummyframe"
      >
        <label htmlFor="file-upload" className="custom-file-upload">
          <FaCloudUploadAlt /> {fileName || "Choose zip file"}
        </label>
        <input
          type="file"
          id="file-upload"
          accept=".zip"
          onChange={handleFileUpload}
          style={{ width: "100%" }}
        />
        <input
          type="submit"
          value="upload"
          style={{ marginLeft: "48px", marginTop: "8px" }}
          onClick={() => setDataUploaded(true)}
        ></input>
      </form>
    </>
  );
};

LargeFileUpload.propTypes = {
  setDataUploaded: PropTypes.any,
};

export default LargeFileUpload;
