import React from "react";
import PropTypes from "prop-types";

const ImageComponent = (props) => {
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

const propTypes = { imageData: PropTypes.object };
ImageComponent.propTypes = propTypes;

export default ImageComponent;
