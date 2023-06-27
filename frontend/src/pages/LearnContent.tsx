import React, { useState, useEffect } from "react";
import MCQuestion from "../features/LearnMod/MCQuestion";
import ImageComponent from "../features/LearnMod/ImageComponent";
import ModulesSideBar from "../features/LearnMod/ModulesSideBar";
import FRQuestion from "../features/LearnMod/FRQuestion";
import { useAppSelector } from "@/common/redux/hooks";
import {
  ContentType,
  ModuleType,
} from "../features/LearnMod/LearningModulesContent";
import Exercise from "../features/LearnMod/Exercise";
import Router from "next/router";
import NavbarMain from "@/common/components/NavBarMain";
import Footer from "@/common/components/Footer";

const LearnContent = () => {
  const [moduleContent, setModuleContent] = useState<ModuleType>(
    JSON.parse(Router.query.moduleContent) as ModuleType
  );
  const [subSection, setSubSection] = useState(
    parseInt(Router.query.subsection)
  );
  const user = useAppSelector((state) => state.currentUser.user);
  console.log("hi");
  console.log(moduleContent.subClasses[1]);
  console.log(subSection);
  useEffect(() => {
    setSubSection(parseInt(Router.query.subsection));
    setModuleContent(JSON.parse(Router.query.moduleContent) as ModuleType);
  }, [Router.query]);

  // moves to previous subsection if there is one
  const onPreviousClick = () => {
    if (subSection !== 0) {
      setSubSection(subSection - 1);
    } else {
      Router.push("/learn");
    }
  };

  //moves to next subsection if there is one
  const onNextClick = () => {
    if (subSection !== moduleContent.subClasses.length - 1) {
      setSubSection(subSection + 1);
    } else {
      Router.push("/learn");
    }
  };

  if (!user) {
    return <></>;
  }
  return (
    <>
      <NavbarMain />
      <div id="header-section">
        <h1 className="headers">{moduleContent.title}</h1>
      </div>
      <div id="learningBody">
        <ModulesSideBar />
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
              if (contentComponent.sectionType === "exercise") {
                return <Exercise />;
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
      <Footer />
    </>
  );
};

export default LearnContent;
