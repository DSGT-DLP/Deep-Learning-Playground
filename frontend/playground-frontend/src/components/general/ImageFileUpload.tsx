import React from "react";
import { FaCloudUploadAlt } from "react-icons/fa";

interface ImageFileUploadProps {
  uploadFile: File | null;
  setUploadFile: (file: File | null) => void;
}

const ImageFileUpload = (props: ImageFileUploadProps) => {
  const { uploadFile, setUploadFile } = props;

  const handleFileUpload = (target: HTMLInputElement) => {
    if (!target.files) throw new Error("No files");
    setUploadFile(target.files[0] ? target.files[0] : null);
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
          <FaCloudUploadAlt />{" "}
          {uploadFile?.name.substring(0, 20) || "Choose jpg file"}
        </label>
        <input
          type="file"
          name="file"
          id="file-upload"
          accept=".jpg"
          onChange={e => handleFileUpload(e.target)}
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


export default ImageFileUpload;
