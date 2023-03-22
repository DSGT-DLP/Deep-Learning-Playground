import React, { useEffect, useState } from "react";
import PropTypes from "prop-types";
import { useNavigate } from "react-router-dom";

interface InfoType {
  moduleId: number;
  points: number;
  subClasses: { title: string }[];
  sectionID: number;
  title: string;
}
const ClassCard = (props: {
  info: InfoType;
  moduleProgress: { modulePoints: number };
}) => {
  useEffect(() => {
    console.log(props);
  }, []);
  const [pointsEarned, setPointsEarned] = useState(0);

  const points = props.info.points;

  // get updated user progress
  useEffect(() => {
    if (props.moduleProgress != null) {
      setPointsEarned(props.moduleProgress.modulePoints);
    }
  });

  let completed = false;
  if (pointsEarned === points) {
    completed = true;
  }

  const navigate = useNavigate();

  const sectionSpec = {
    moduleContent: props.info,
    subsection: 0,
  };

  return (
    <div
      className="class"
      style={completed ? { border: "3px solid green" } : {}}
    >
      <div className="classHeader">
        <h3 id="classTitle"> {props.info.title} </h3>
        <div className="circleWrap">
          <div
            className="circleFill"
            style={
              completed
                ? {
                    background: ` conic-gradient(green ${
                      (360 * pointsEarned) / points
                    }deg, #ededed 0deg)`,
                  }
                : {
                    background: ` conic-gradient(var(--primary) ${
                      (360 * pointsEarned) / points
                    }deg, #ededed 0deg)`,
                  }
            }
          >
            <div className="circleInFill">
              <span className="textInCircle">
                {Math.floor((pointsEarned / points) * 100)}%
              </span>
            </div>
          </div>
        </div>
      </div>
      <div className="classBody">
        <div className="classBodyLeft">
          {props.info.subClasses.map((subClass, index) => {
            return (
              <p id="classBodyText" key={index}>
                {subClass.title}
              </p>
            );
          })}
        </div>
        <div className="classBodyRight">
          {props.info.subClasses.map((subClass, index) => {
            return (
              <div className="progressBar" key={index} id="subClassBar">
                <div
                  className="progressContainer"
                  id="subClassBar"
                  style={completed ? { border: "3px solid green" } : {}}
                >
                  <div
                    className="progressFill"
                    style={
                      completed
                        ? {
                            backgroundColor: "green",
                            width: `${pointsEarned}%`,
                          }
                        : { width: `${pointsEarned}%` }
                    }
                  ></div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <div className="classFooter">
        <button
          id="classBtn"
          onClick={() => navigate("/LearnContent", { state: sectionSpec })}
          style={completed ? { border: "3px solid green" } : {}}
        >
          {completed ? "Completed" : "Start"}
        </button>
      </div>
    </div>
  );
};

const propTypes = {
  info: PropTypes.object,
  user: PropTypes.object,
  moduleProgress: PropTypes.object,
};
ClassCard.propTypes = propTypes;

export default ClassCard;
