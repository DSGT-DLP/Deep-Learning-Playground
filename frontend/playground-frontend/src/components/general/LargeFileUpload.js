import React, { useEffect, useState } from "react";

const { Dropzone } = require("dropzone");


const LargeFileUpload = () => {

  

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
        console.log("init");
      }
    }
    Dropzone.options.dropper.init();
    console.log(Dropzone.options.dropper);
  });

  return (
    <>
      <iframe name="dummyframe" id="dummyframe" style={{'display':"none"}}></iframe>
      <form action="/upload" enctype='multipart/form-data' method='POST' target="dummyframe">
        <input type="file" name="file"></input>
        <input type="submit" value="upload"></input>
      </form>
    </>
  );
};

export default LargeFileUpload;
