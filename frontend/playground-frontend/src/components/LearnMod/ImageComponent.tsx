import React from "react";
interface ImageData {
  path: string;
  caption: string;
  licenseLink: string;
  attribution: string;
}

const ImageComponent = (props: { imageData: ImageData }) => {
  return (
    <div id="imageContainer">
      <img
        className="learnImg"
        src={props.imageData.path}
        alt={props.imageData.caption}
      ></img>
      <p className="captionText">{props.imageData.caption}</p>
      <a className="attributionLink" href={props.imageData.licenseLink}>
        {props.imageData.attribution}
      </a>
    </div>
  );
};

export default ImageComponent;
