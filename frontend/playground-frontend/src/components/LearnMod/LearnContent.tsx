import React, { useState, useEffect } from "react";
import MCQuestion from "./MCQuestion";
import ImageComponent from "./ImageComponent";
import { useLocation } from "react-router-dom";
import { useNavigate } from "react-router-dom";
import ModulesSideBar from "./ModulesSideBar";
import FRQuestion from "./FRQuestion";
import { useAppSelector } from "../../redux/hooks";
import { UserState } from "../../redux/userLogin";
import { ContentType, ModuleType } from "./LearningModulesContent";

const LearnContent = () => {
  const navigate = useNavigate();

  const location = useLocation();

  const [moduleContent, setModuleContent] = useState<ModuleType>(
    location.state.moduleContent
  );
  const [subSection, setSubSection] = useState(location.state.subsection);
  const user = useAppSelector<UserState>((state) => state.currentUser);
  useEffect(() => {
    setSubSection(location.state.subsection);
    setModuleContent(location.state.moduleContent);
  }, [location.state]);

  // moves to previous subsection if there is one
  const onPreviousClick = () => {
    if (subSection !== 0) {
      setSubSection(subSection - 1);
    } else {
      navigate("/learn-mod");
    }
  };

  //moves to next subsection if there is one
  const onNextClick = () => {
    if (subSection !== moduleContent.subClasses.length - 1) {
      setSubSection(subSection + 1);
    } else {
      navigate("/learn-mod");
    }
  };

  return (
    <>
      <div id="header-section">
        <h1 className="headers">{moduleContent.title}</h1>
      </div>
      <div id="learningBody">
        <div className="sideBar">
          <ModulesSideBar />
        </div>
        <div className="learningContentDiv">
          <h2>{moduleContent.subClasses[subSection].title}</h2>
          {moduleContent.subClasses[subSection].content.map(
            (contentComponent: ContentType, index: number) => {
              if (contentComponent.sectionType === "text") {
                return (
                  <p className="contentParagraph" key={index}>
                    {(contentComponent as ContentType<"text">).content}
                  </p>
                );
              }

              if (contentComponent.sectionType === "heading1") {
                return (
                  <h5 className="heading1" key={index}>
                    {(contentComponent as ContentType<"text">).content}
                  </h5>
                );
              }

              if (contentComponent.sectionType === "image") {
                return (
                  <ImageComponent
                    key={index}
                    imageData={contentComponent as ContentType<"image">}
                  />
                );
              }

              if (contentComponent.sectionType === "mcQuestion") {
                return (
                  <MCQuestion
                    key={index}
                    user={user}
                    questionObject={
                      contentComponent as ContentType<"mcQuestion">
                    }
                    moduleID={moduleContent.moduleID}
                    sectionID={moduleContent.subClasses[subSection].sectionID}
                  />
                );
              }

              if (contentComponent.sectionType === "frQuestion") {
                return (
                  <FRQuestion
                    key={index}
                    user={user}
                    questionObject={
                      contentComponent as ContentType<"frQuestion">
                    }
                    moduleID={moduleContent.moduleID}
                    sectionID={moduleContent.subClasses[subSection].sectionID}
                  />
                );
              }
            }
          )}
        </div>
      </div>
      <div id="subsectionChangeContainer">
        <button className="class" onClick={onPreviousClick}>
          Previous
        </button>
        <button className="class" onClick={onNextClick}>
          Next
        </button>
      </div>
    </>
  );
};

export default LearnContent;
