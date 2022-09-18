import React from "react";
import { PropTypes } from "prop-types";
import { FaCloudUploadAlt } from "react-icons/fa";

const LargeFileUpload = (props) => {
  const { uploadFile, setUploadFile } = props;

  const handleFileUpload = (e) => {
    e.preventDefault();
    setUploadFile(e.target.files[0] ? e.target.files[0] : null);
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
          <FaCloudUploadAlt /> {uploadFile?.name.substring(0, 20) || "Choose zip file"}
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
          hidden
        ></input>
      </form>
    </>
  );
};

LargeFileUpload.propTypes = {
  uploadFile: PropTypes.object,
  setUploadFile: PropTypes.func.isRequired,
};

export default LargeFileUpload;
