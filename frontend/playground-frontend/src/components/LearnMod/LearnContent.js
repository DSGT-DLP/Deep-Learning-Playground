import React, { useState, useEffect } from 'react';
import MCQuestion from './MCQuestion';
import FRQuestion from './FRQuestion';
import ImageComponent from './ImageComponent';
import { useLocation } from 'react-router-dom';
import { useNavigate } from 'react-router-dom';
import { auth } from '../../firebase';

const LearnContent = () => {

    const navigate = useNavigate();
  
    const location = useLocation();
    
    let moduleContent = location.state;
    
    //current user logged in
    const [user, setUser] = useState(null);

    //current subsection being displayed
    const [subSection, setSubSection] = useState(0);

    // check if logged in
    useEffect(() => {

      auth.onAuthStateChanged((userLogged) => {

          if (userLogged) {

              setUser(userLogged);

          } else {

              navigate("/login");

          }
  
      });

    });

    // moves to previous subsection if there is one
    const onPreviousClick = () => {

      if (subSection !== 0) {

        setSubSection(subSection - 1);

      }

    };
    
    //moves to next subsection if there is one
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
      <div id='learningBody'>
        <div className="learningContentDiv">
            <h2>{moduleContent.subClasses[subSection].title}</h2>
            {
              moduleContent.subClasses[subSection].content.map((contentComponent, index) => {

                if (contentComponent.sectionType === "text") {

                  return (
                    <p className="contentParagraph" key={index}>{contentComponent.content}</p>
                  );

                }

                if (contentComponent.sectionType === "image") {

                  return (
                    <ImageComponent key={index} imageData={contentComponent}/>
                  );

                }

                if (contentComponent.sectionType === "mcQuestion") {

                  return (
                    <MCQuestion key={index} 
                    user={user}
                    questionObject={contentComponent} 
                    moduleID={moduleContent.moduleID} 
                    sectionID={moduleContent.subClasses[subSection].sectionID}/>
                  );

                }

                if (contentComponent.sectionType === "frQuestion") {

                  return (
                    <FRQuestion key={index} 
                    user={user}
                    questionObject={contentComponent} 
                    moduleID={moduleContent.moduleID} 
                    sectionID={moduleContent.subClasses[subSection].sectionID}/>
                  );

                }

              })
            }

        </div>
      </div>
      <button onClick={onPreviousClick}>Previous</button>
      <button onClick={onNextClick}>Next</button>
        </>
    );

};

export default LearnContent;
