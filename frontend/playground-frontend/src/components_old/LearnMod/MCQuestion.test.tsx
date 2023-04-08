import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom";
import MCQuestion from "./MCQuestion";
import { sendToBackend } from "../helper_functions/TalkWithBackend";
import { CTypes, ContentType } from "../../common/components/LearningModulesContent";

jest.mock("../helper_functions/TalkWithBackend");

function getMockUser() {
    return {
      uid: "123",
      email: "example@example.com",
      displayName: "example",
      emailVerified: true,
    };
}

describe("MCQuestion_function", () => {
  // Tests that when the user selects the correct answer, progress is updated and feedback message is displayed. tags: [happy path]
  it("test_question_submit_correct_answer", async () => {
    const mockUser = getMockUser();
    const mockQuestionObject: ContentType<"mcQuestion"> = {
        sectionType: "mcQuestion" as CTypes,
        questionID: 1,
        question: "What is 2+2?",
        answerChoices: ["3", "4", "5", "6"],
        correctAnswer: 1, 
        path: null,
        caption: null,
        attribution: null,
        licenseLink: null,
        content: null,
        minAccuracy: null,
        answer: undefined
    };
    const mockProps = {
        moduleID: 1,
        questionObject: mockQuestionObject,
        sectionID: 1,
        user: mockUser
    };
    render(
      <MCQuestion {...mockProps}
      />
    );

    //seems like correct answer is based off of index
    const select = screen.getByDisplayValue("1"); 
    const submitButton = screen.getByText("Submit Answer");
    userEvent.click(select);
    userEvent.click(submitButton);
    await waitFor(() =>
      expect(screen.getByText("That is correct!")).toBeInTheDocument()
    );

    expect(sendToBackend).toHaveBeenCalledWith("updateUserProgressData", {
      uid: mockUser.uid,
      moduleID: mockProps.moduleID,
      sectionID: mockProps.sectionID,
      questionID: mockProps.questionObject.questionID,
    });
  });

  // Tests that when the user selects an incorrect answer, progress is not updated and feedback message is displayed. tags: [happy path]
  it("test_question_submit_incorrect_answer", async () => {
    const mockUser = getMockUser();
    const mockQuestionObject: ContentType<"mcQuestion"> = {
        sectionType: "mcQuestion" as CTypes,
        questionID: 1,
        question: "What is 2+2?",
        answerChoices: ["3", "4", "5", "6"],
        correctAnswer: 1, 
        path: null,
        caption: null,
        attribution: null,
        licenseLink: null,
        content: null,
        minAccuracy: null,
        answer: undefined
    };
    const mockProps = {
        moduleID: 1,
        questionObject: mockQuestionObject,
        sectionID: 1,
        user: mockUser
    };
    render(
      <MCQuestion {...mockProps}
      />
    );

    //seems like correct answer is based off of index
    const select = screen.getByDisplayValue("0"); 
    const submitButton = screen.getByText("Submit Answer");
    userEvent.click(select);
    userEvent.click(submitButton);
    await waitFor(() =>
      expect(screen.getByText("Sorry, that is incorrect")).toBeInTheDocument()
    );

    await waitFor(() => expect(sendToBackend).not.toHaveBeenCalled());
  });

  // Tests that when the user does not select an answer, a prompt message is displayed. tags: [happy path]
  it("test_question_submit_unanswered", async () => {
    const mockUser = getMockUser();
    const mockQuestionObject: ContentType<"mcQuestion"> = {
        sectionType: "mcQuestion" as CTypes,
        questionID: 1,
        question: "What is 2+2?",
        answerChoices: ["3", "4", "5", "6"],
        correctAnswer: 1, 
        path: null,
        caption: null,
        attribution: null,
        licenseLink: null,
        content: null,
        minAccuracy: null,
        answer: undefined
    };
    const mockProps = {
        moduleID: 1,
        questionObject: mockQuestionObject,
        sectionID: 1,
        user: mockUser
    };
    render(
      <MCQuestion {...mockProps}
      />
    );
    userEvent.click(screen.getByText("Submit Answer"));
    expect(screen.getByText("Please select an answer")).toBeInTheDocument();
  });

  // Tests that when the subsection changes, the state variables are reset and a new question is displayed. tags: [general behavior]
  it("test_question_reset_on_subsection_change", async () => {
    const mockUser = getMockUser();
    const mockQuestionObject: ContentType<"mcQuestion"> = {
        sectionType: "mcQuestion" as CTypes,
        questionID: 1,
        question: "What is 2+2?",
        answerChoices: ["3", "4", "5", "6"],
        correctAnswer: 1, 
        path: null,
        caption: null,
        attribution: null,
        licenseLink: null,
        content: null,
        minAccuracy: null,
        answer: undefined
    };
    const mockProps = {
        moduleID: 1,
        questionObject: mockQuestionObject,
        sectionID: 1,
        user: mockUser
    };
    const {rerender} = render(
      <MCQuestion
        {...mockProps}
      />
    );
    userEvent.click(screen.getByText("Submit Answer"));
    rerender(
      <MCQuestion
        moduleID={1}
        questionObject={{
          questionID: 2,
          question: "What is 3+3?",
          answerChoices: ["5", "6", "7", "8"],
          correctAnswer: 1, //index
          sectionType: "mcQuestion",
          path: null,
          caption: null,
          attribution: null,
          licenseLink: null,
          content: null,
          minAccuracy: null,
          answer: undefined

        }}
        sectionID={2}
        user={mockUser}
      />
    );
    expect(screen.getByText("Question")).toBeInTheDocument();
    expect(screen.getByText("What is 3+3?")).toBeInTheDocument();
    expect(screen.getByText("Submit Answer")).toBeInTheDocument();
  });
});
