import React, { useState } from "react";
import TitleText from "./mini_components/TitleText";

const CodeSnippet = (props) => {

    const { backendResponse, layers } = props;  

    if (!backendResponse?.success) {
        return (
          backendResponse?.message || (
            <p style={{ textAlign: "center" }}>There are no records to display</p>
          )
        );
      }
    
    return (
        //Just an example output since I can't print objects out
        //TODO: Make this print out the entire actual code by looking at the layers
        <textarea
            style={{ width:"50%" }}
            value={layers[0].display_name + "\n" + "test"}
        />
    );
};

export default CodeSnippet;


