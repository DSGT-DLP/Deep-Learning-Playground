import ClassCard from "../features/LearnMod/ClassCard";
import getContent from "../features/LearnMod/LearningModulesContent";
import { useAppSelector } from "@/common/redux/hooks";
import NavbarMain from "@/common/components/NavBarMain";
import Footer from "@/common/components/Footer";
import React from "react";
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
      <NavbarMain />
      <div id="header-section">
        <h1 className="headers">Your learning modules, {user.displayName}</h1>
      </div>
      <div id="learningBody">
        <div className="classes">
          {content.map((lesson, index) => {
            //const moduleProgress = user.userProgressData[lesson.moduleID];
            console.log(lesson);
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
      <Footer />
    </>
  );
};

export default LearnMod;
