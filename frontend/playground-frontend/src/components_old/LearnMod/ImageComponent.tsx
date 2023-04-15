import React from "react";
import { ContentType } from "./LearningModulesContent";

const ImageComponent = (props: { imageData: ContentType<"image"> }) => {
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
