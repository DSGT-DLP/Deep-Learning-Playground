import React, { useEffect } from "react";
import { PropTypes } from "prop-types";

const { Dropzone } = require("dropzone");

const LargeFileUpload = (props) => {

  const {setDataUploaded } = props;

  const script = document.createElement("script");

  script.src = "https://cdnjs.cloudflare.com/ajax/libs/dropzone/5.9.3/dropzone-amd-module.min.js";
  script.async = true;

  document.body.appendChild(script);

  
  useEffect(() => {
    Dropzone.autoDiscover = false;
    Dropzone.options.dropper = {
      paramName: 'file',
      chunking: true,
      forceChunking: true,
      url: '/upload',
      maxFilesize: 5000, // megabytes
      chunkSize: 1000000, // bytes
      init: function() {
        ;
      }
    }
    Dropzone.options.dropper.init();
  });

  return (
    <>
      <iframe name="dummyframe" id="dummyframe" style={{'display':"none"}}></iframe>
      <form action="/upload" encType='multipart/form-data' method='POST' target="dummyframe" >
        <input type="file" name="file" accept=".zip" style={{marginLeft: "40px"}}></input>
        <input type="submit" value="upload" style={{marginLeft: "43px"}} onClick={()=> setDataUploaded(true)}></input>
      </form>
    </>
  );
};

LargeFileUpload.propTypes = {
  setDataUploaded: PropTypes.any,
}

export default LargeFileUpload;