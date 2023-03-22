import React from "react";
import ClassCard from "./ClassCard";
import getContent from "./LearningModulesContent";
import { useAppSelector } from "../../redux/hooks";
//import { fetchUserProgressData } from "../../redux/userLogin";

const LearnMod = () => {
  //module content
  const content = getContent.modules;

  const user = useAppSelector((state) => state.currentUser);
  //const dispatch = useAppDispatch();
  /*
  useEffect(() => {
    auth.onAuthStateChanged((user) => {
      if (user) {
        dispatch(fetchUserProgressData());
      }
    });
  }, []);
  useEffect(() => {
    console.log(user.userProgressData);
  }, []);*/

  return (
    <>
      <div id="header-section">
        <h1 className="headers">Your learning modules, {user.displayName}</h1>
      </div>
      <div id="learningBody">
        <div className="classes">
          {content.map((lesson, index) => {
            //const moduleProgress = user.userProgressData[lesson.moduleID];

            return (
              <ClassCard
                user={user}
                info={lesson}
                key={index}
                moduleProgress={{
                  modulePoints: 0,
                }}
              />
            );
          })}
        </div>
      </div>
    </>
  );
};

export default LearnMod;
