import React from "react";
import ClassCard from "../common/components/ClassCard";
import getContent from "../common/components/LearningModulesContent";
import { useAppSelector } from "../common/redux/hooks";
//import { fetchUserProgressData } from "../../redux/userLogin";

const LearnMod = () => {
  //module content
  const content = getContent.modules;

  const user = useAppSelector((state) => state.currentUser.user);
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

  if (!user) {
    return <></>;
  }
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
