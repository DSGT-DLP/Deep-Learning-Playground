import React, { useState } from 'react';
import MCQuestion from './MCQuestion';
import { useLocation } from 'react-router-dom';

const LearnContent = () => {

    const location = useLocation();
    
    let moduleContent = location.state;
    
    const [subSection, setSubSection] = useState(0);

    const onPreviousClick = () => {

      if (subSection !== 0) {

        setSubSection(subSection - 1);

      }

    };
    
    const onNextClick = () => {

      if (subSection !== moduleContent.subClasses.length - 1) {

        setSubSection(subSection + 1);

      }

    };

    return (
        <>
      <div id="header-section">
            <h1 className="headers">{moduleContent.title}</h1>
      </div>
      <div id="learningContentDiv">
          <h1>{moduleContent.subClasses[subSection].title}</h1>
          {
            moduleContent.subClasses[subSection].content.map((contentComponent, index) => {

              if (contentComponent.sectionType === "text") {

                return (
                  <p key={index}>{contentComponent.content}</p>
                );

              }

              if (contentComponent.sectionType === "mcQuestion") {

                return (
                  <MCQuestion/>
                );

              }

            })
          }

      </div>
      <button onClick={onPreviousClick}>Previous</button>
      <button onClick={onNextClick}>Next</button>
        </>
    );

};

export default LearnContent;
