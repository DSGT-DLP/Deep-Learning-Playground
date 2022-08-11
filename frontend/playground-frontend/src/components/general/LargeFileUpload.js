import React, { useState } from "react";
import { PropTypes } from "prop-types";
import { FaCloudUploadAlt } from "react-icons/fa";

const LargeFileUpload = (props) => {
  const { setDataUploaded } = props;
  const [fileName, setFileName] = useState();

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    setFileName(file.name.substring(0, 20));
  };

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
          name="file"
          id="file-upload"
          accept=".zip"
          onChange={handleFileUpload}
          style={{ width: "100%" }}
        />
        <input
          type="submit"
          value="Upload"
          id="fileUploadInput"
          style={{ marginLeft: "48px", marginTop: "8px" }}
          onClick={() => setDataUploaded(true)}
          hidden
        ></input>
      </form>
    </>
  );
};

LargeFileUpload.propTypes = {
  setDataUploaded: PropTypes.func.isRequired,
};

export default LargeFileUpload;
