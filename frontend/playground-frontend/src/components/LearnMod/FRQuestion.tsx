import React, { useEffect, useState } from "react";
import PropTypes from "prop-types";
import { sendToBackend } from "../helper_functions/TalkWithBackend";
import { User } from "firebase/auth";

interface QuestionType {
  answer: number;
  question: string;
  questionID: string;
  sectionType: string;
}
const FRQuestion = (props: {
  moduleID: string;
  questionObject: QuestionType;
  sectionID: number;
  user: User;
}) => {
  const [answeredCorrect, setAnsweredCorrect] = useState(false);
  const [answeredIncorrect, setAnsweredIncorrect] = useState(false);
  const [unanswered, setUnanswered] = useState(false);
  useEffect(() => {
    console.log(props);
  }, [answeredCorrect]);

  // function that makes call to backend to update user progress
  async function updateUserProgress() {
    const requestData = {
      uid: props.user.uid,
      moduleID: props.moduleID,
      sectionID: props.sectionID,
      questionID: props.questionObject.questionID,
    };

    sendToBackend("updateUserProgressData", requestData);
  }

  // run when submit button on question is pressed
  const questionSubmit = () => {
    if ((document.getElementById("frInput") as HTMLInputElement).value === "") {
      setUnanswered(true);
      setAnsweredCorrect(false);
      setAnsweredIncorrect(false);
    } else {
      const answer = parseInt(
        (document.getElementById("frInput") as HTMLInputElement).value
      );

      if (answer === props.questionObject.answer) {
        setAnsweredCorrect(true);
        setAnsweredIncorrect(false);
        setUnanswered(false);

        updateUserProgress();
      } else {
        setAnsweredCorrect(false);
        setAnsweredIncorrect(true);
        setUnanswered(false);
      }
    }
  };

  //resets question when subsection changes
  useEffect(() => {
    setAnsweredCorrect(false);
    setAnsweredIncorrect(false);
    setUnanswered(false);
  }, [props]);

  return (
    <div className="class">
      <h3 id="classTitle">Question</h3>
      <h6>{props.questionObject.question}</h6>
      <input id="frInput" type="number"></input>
      <button className="submitButton" onClick={questionSubmit}>
        Submit Answer
      </button>
      {answeredCorrect ? (
        <h6 style={{ color: "green" }}>That is correct!</h6>
      ) : null}
      {answeredIncorrect ? (
        <h6 style={{ color: "red" }}>Sorry, that is incorrect</h6>
      ) : null}
      {unanswered ? (
        <h6 style={{ color: "orange" }}>Please type an answer</h6>
      ) : null}
    </div>
  );
};

const propTypes = {
  user: PropTypes.object,
  questionObject: PropTypes.object,
  moduleID: PropTypes.string,
  sectionID: PropTypes.number,
};
FRQuestion.propTypes = propTypes;

export default FRQuestion;
