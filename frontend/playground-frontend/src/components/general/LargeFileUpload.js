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
      <link rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/dropzone/5.4.0/min/dropzone.min.css" />

      <link rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/dropzone/5.4.0/min/basic.min.css" />
      <form method="POST" action='/upload' className="dropzone dz-clickable"
        id="dropper" encType="multipart/form-data">
      </form>
      <div className="myId"></div>
    </>
  );
};

export default LargeFileUpload;
