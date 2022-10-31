import React, { useState } from 'react';
import { useLocation } from 'react-router-dom';

const LearnContent = () => {

    const location = useLocation();
    
    let moduleContent = location.state;
    
    const [subSection] = useState(0);

    return (
        <>
      <div id="header-section">
            <h1 className="headers">{moduleContent.title}</h1>
      </div>
      <div id="learningContentDiv">
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
        </>
    );

};

export default LearnContent;
