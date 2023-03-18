import React, { useEffect, useState } from "react";
import ClassCard from "./ClassCard";
import getContent from "./LearningModulesContent";
import { auth } from "../../firebase";
import { useNavigate } from "react-router-dom";
import { sendToBackend } from "../helper_functions/TalkWithBackend";

const LearnMod = () => {
  const navigate = useNavigate();

  //module content
  const content = getContent.modules;

  //current logged in user
  const [user, setUser] = useState(null);

  //user progress data (loaded in from database)
  const [userProgressData, setUserProgressData] = useState(null);
  const [userDataFetchInitiated, setUserDataFetchInitiated] = useState(false);

  async function getUserProgress(userLogged) {
    setUserDataFetchInitiated(true);
    sendToBackend("getUserProgressData", userLogged.uid).then((result) => {
      setUserProgressData(result);
    });
  }

  useEffect(() => {
    auth.onAuthStateChanged((userLogged) => {
      if (userLogged) {
        setUser(userLogged);

        if (userProgressData == null && !userDataFetchInitiated) {
          getUserProgress(userLogged);
        }
      } else {
        navigate("/login");
      }
    });
  });

  let lessons = content.map((x) => ({
    title: x.title,
    points: x.points,
    subClasses: x.subsections,
    moduleID: x.moduleID,
  }));

  if (user != null) {
    return (
      <>
        <div id="header-section">
          <h1 className="headers">Your learning modules, {user.displayName}</h1>
        </div>

        <div id="learningBody">
          <div className="classes">
            {lessons.map((lesson, index) => {
              let moduleProgress = null;

              if (userProgressData != null) {
                moduleProgress = userProgressData[lesson.moduleID];
              }

              return (
                <ClassCard
                  user={user}
                  info={lesson}
                  key={index}
                  moduleProgress={moduleProgress}
                />
              );
            })}
          </div>
        </div>
      </>
    );
  }
  return <></>;
};

export default LearnMod;
