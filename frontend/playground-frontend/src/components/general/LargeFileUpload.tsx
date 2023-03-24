import React from "react";
import { FaCloudUploadAlt } from "react-icons/fa";

interface LargeFileUploadProps {
  uploadFile: File | null;
  setUploadFile: React.Dispatch<React.SetStateAction<File | null>>;
}
const LargeFileUpload = (props: LargeFileUploadProps) => {
  const { uploadFile, setUploadFile } = props;

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
          <FaCloudUploadAlt />{" "}
          {uploadFile?.name.substring(0, 20) || "Choose zip file"}
        </label>
        <input
          type="file"
          name="file"
          id="file-upload"
          accept=".zip"
          onChange={(e) => {
            e.preventDefault();
            setUploadFile(e.target?.files?.[0] ? e.target.files[0] : null);
          }}
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

export default LargeFileUpload;
