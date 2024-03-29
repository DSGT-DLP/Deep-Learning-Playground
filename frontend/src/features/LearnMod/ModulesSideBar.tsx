import React from "react";
import getContent from "./LearningModulesContent";
import Router from "next/router";
import { useState } from "react";

const ModulesSideBar = () => {
  const content = getContent.modules;

  const [sideBarOpen, setSideBarOpen] = useState(true);
  const [sideBarToggleText, setSideBarToggleText] = useState("<<");
  const [sideBarToggleDisplay, setSideBarToggleDisplay] = useState("block");

  const onSideBarToggle = () => {
    if (sideBarOpen) {
      setSideBarOpen(false);
      setSideBarToggleText(">>");
      setSideBarToggleDisplay("none");
    } else {
      setSideBarOpen(true);
      setSideBarToggleText("<<");
      setSideBarToggleDisplay("block");
    }
  };

  return (
    <div className="sideBar">
      <div style={{ display: sideBarToggleDisplay }}>
        <h2 style={{ color: "white" }}>Modules</h2>
        <ul style={{ padding: 0 }}>
          {content.map((lesson, index) => {
            return (
              <li className="sideBarModule" key={index}>
                <p style={{ color: "white" }}>{lesson.title}</p>
                <ul>
                  {lesson.subClasses.map((subsection, index2) => {
                    const sectionSpec = {
                      moduleContent: JSON.stringify(lesson),
                      subsection: index2,
                    };

                    return (
                      <li
                        className="sideBarSubsection"
                        onClick={() =>
                          Router.push(
                            {
                              pathname: "/LearnContent",
                              query: sectionSpec,
                            },
                            "/LearnContent"
                          )
                        }
                        key={index2}
                        style={{ color: "white" }}
                      >
                        {subsection.title}
                      </li>
                    );
                  })}
                </ul>
              </li>
            );
          })}
        </ul>
      </div>
      <button onClick={onSideBarToggle} id="sideBarToggleButton">
        {sideBarToggleText}
      </button>
    </div>
  );
};

export default ModulesSideBar;
