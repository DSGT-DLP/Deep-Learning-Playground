import React, { useEffect, useState } from "react";
//import { sendToBackend } from "../helper_functions/TalkWithBackend";
import { UserType } from "../redux/userLogin";
import { ContentType } from "../../common/components/LearningModulesContent";

export interface MCQuestionType {
  answerChoices: string[];
  correctAnswer: number;
  question: string;
  questionID: string;
  sectionType: string;
}
interface MCQuestionProps {
  moduleID: number;
  questionObject: ContentType<"mcQuestion">;
  sectionID: number;
  user: UserType;
}
const MCQuestion = (props: MCQuestionProps) => {
  const [answeredCorrect, setAnsweredCorrect] = useState(false);
  const [answeredIncorrect, setAnsweredIncorrect] = useState(false);
  const [unanswered, setUnanswered] = useState(false);

  // function that makes call to backend to update user progress
  async function updateUserProgress() {
    const requestData = {
      uid: props.user.uid,
      moduleID: props.moduleID,
      sectionID: props.sectionID,
      questionID: props.questionObject.questionID,
    };

    //sendToBackend("updateUserProgressData", requestData);
  }

  // run when submit button on question is pressed
  const questionSubmit = () => {
    if (
      document.querySelector(
        'input[name="' + props.questionObject.questionID + '"]:checked'
      ) == null
    ) {
      setUnanswered(true);
      setAnsweredCorrect(false);
      setAnsweredIncorrect(false);
    } else {
      const answer = parseInt(
        (
          document.querySelector(
            'input[name="' + props.questionObject.questionID + '"]:checked'
          ) as HTMLInputElement
        ).value
      );

      if (answer === props.questionObject.correctAnswer) {
        setAnsweredCorrect(true);
        setAnsweredIncorrect(false);

        updateUserProgress();
      } else {
        setAnsweredCorrect(false);
        setAnsweredIncorrect(true);
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
      {props.questionObject.answerChoices.map((answer, index) => {
        return (
          <div key={index}>
            <input
              type="radio"
              value={index}
              name={props.questionObject.questionID.toString()}
            />
            {answer}
          </div>
        );
      })}
      <button onClick={questionSubmit}>Submit Answer</button>
      {answeredCorrect ? (
        <h6 style={{ color: "green" }}>That is correct!</h6>
      ) : null}
      {answeredIncorrect ? (
        <h6 style={{ color: "red" }}>Sorry, that is incorrect</h6>
      ) : null}
      {unanswered ? (
        <h6 style={{ color: "orange" }}>Please select an answer</h6>
      ) : null}
    </div>
  );
};

export default MCQuestion;
