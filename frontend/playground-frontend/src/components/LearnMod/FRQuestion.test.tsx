import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom";
import FRQuestion from "./FRQuestion";
import { sendToBackend } from "../helper_functions/TalkWithBackend";
import { CTypes } from "./LearningModulesContent";

jest.mock("../helper_functions/TalkWithBackend");

function getMockUser() {
  return {
    uid: "123",
    email: "example@example.com",
    displayName: "example",
    emailVerified: true,
  };
}
describe("FRQuestion_function", () => {
  // Tests that the component updates the progress and displays a success message when the user inputs the correct answer. tags: [happy path]
  it("test_correct_answer", async () => {
    const mockUser = getMockUser();
    const mockQuestionObject = {
      sectionType: "frQuestion" as CTypes,
      questionID: 1,
      question: "What is 2+2?",
      answer: 4,
      answerChoices: null,
      correctAnswer: null,
      path: null,
      caption: null,
      attribution: null,
      licenseLink: null,
      content: null,
      minAccuracy: null,
    };
    const mockProps = {
      moduleID: 1,
      questionObject: mockQuestionObject,
      sectionID: 1,
      user: mockUser,
    };

    render(<FRQuestion {...mockProps} />);
    const input = screen.getByTestId("frInput");
    const submitButton = screen.getByText("Submit Answer");

    userEvent.type(input, "4");
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

  // Tests that the component displays an error message when the user submits an empty input. tags: [edge case]
  it("test_empty_input", async () => {
    const mockUser = getMockUser();
    const mockQuestionObject = {
      sectionType: "frQuestion" as CTypes,
      questionID: 1,
      question: "What is 2+2?",
      answer: 4,
      answerChoices: null,
      correctAnswer: null,
      path: null,
      caption: null,
      attribution: null,
      licenseLink: null,
      content: null,
      minAccuracy: null,
    };
    const mockProps = {
      moduleID: 1,
      questionObject: mockQuestionObject,
      sectionID: 1,
      user: mockUser,
    };

    render(<FRQuestion {...mockProps} />);
    const input = screen.getByTestId("frInput");
    const submitButton = screen.getByText("Submit Answer");

    userEvent.type(input, "");
    userEvent.click(submitButton);

    await waitFor(() =>
      expect(screen.getByText("Please type an answer")).toBeInTheDocument()
    );
  });

  // Tests that the component resets its state when the props change. tags: [edge case]
  it("test_props_change", async () => {
    const mockUser = getMockUser();
    const mockQuestionObject = {
      sectionType: "frQuestion" as CTypes,
      questionID: 1,
      question: "What is 2+2?",
      answer: 4,
      answerChoices: null,
      correctAnswer: null,
      path: null,
      caption: null,
      attribution: null,
      licenseLink: null,
      content: null,
      minAccuracy: null,
    };
    const mockProps = {
      moduleID: 1,
      questionObject: mockQuestionObject,
      sectionID: 1,
      user: mockUser,
    };
    const { rerender } = render(<FRQuestion {...mockProps} />);
    const input = screen.getByTestId("frInput");

    userEvent.type(input, "5");
    rerender(<FRQuestion {...mockProps} />);

    await waitFor(() => expect(input).toHaveValue(5));
  });
});
