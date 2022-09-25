import React, { useState } from 'react';
import { useLocation } from 'react-router-dom';

const LearnContent = () => {

    const location = useLocation();
    
    let moduleContent = location.state;
    
    const [subSection, setSubSection] = useState(0);

    return (
        <>
      <div id="header-section">
            <h1 className="headers">{moduleContent.title}</h1>
      </div>
      <div id="learningContentDiv">
            <h2>{moduleContent.subClasses[subSection].title}</h2>
          {
            moduleContent.subClasses[subSection].content.map((contentComponent, index) => {

              if (contentComponent.sectionType === "text") {

                return (
                  <p key={index}>{contentComponent.content}</p>
                );

              }

            })
          }
      </div>
      <div>
        <button onClick={() => {if (subSection > 0) {setSubSection(subSection - 1);}}}>Previous</button>
        <button onClick={() => {if (subSection < moduleContent.subClasses.length - 1) {setSubSection(subSection + 1);}}}>Next</button>
      </div>   
        </>
    );

};

export default LearnContent;
