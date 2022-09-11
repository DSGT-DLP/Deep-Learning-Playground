import React from "react";
import PropTypes from "prop-types";
import uploader from '../../assets/data_upload.gif';
import trainer from '../../assets/training.png';
import wrapup from '../../assets/wrapup.gif';
const LoadingScreen = (props) => {
    const {  upload, progress } = props;
    console.log(upload);


    const main = {
        width: "80%",
        margin: "40px auto"
      };
      
      /* Style the overlay */
    const overlay = {
        height: "100%",
        width: "100%",
        position: "fixed",
        zIndex: 1,
        top: 0,
        left: 0,
        backgroundColor: "rgba(1, 0, 0 , 0.95)",
        transition: "10s",
        overflowX: "hidden"
      };
      
      const overlayContent = {
        position: "relative",
        top: "25%",
        width: "100%",
        textAlign: "center",
        marginTop: "30px",
        color: "white"

      };
      
      const img = {
        display: 'block',
        marginLeft: 'auto',
        marginRight: 'auto'
      };
      

    return (
        <>
            <>
             {/* Main content */}
             <div style={main}></div>
            <h1> {progress} </h1>
           {/* The training overlay */}
           <div style={overlay}>
             <div style={overlayContent}>
        {upload === false ? (
            <>
            <img src={uploader} style={img} />
            <h1>Uploading data</h1>
            </>
        ):
         50 > progress  ? (
         <>   
         <img src={trainer} style={img} />
        <h1> Training Data ...</h1>

        </>
        ): 50 < progress  ? (
            <>
            <img src={wrapup} style={img} />
            <h1>Almost There!</h1>
            </>
        ):  null}
             </div>
           </div>
        </>
        
        </>
        
    );  
  };
  LoadingScreen.propTypes = {
    progress: PropTypes.number,
    upload: PropTypes.bool
  };

export default LoadingScreen;
