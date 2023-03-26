import React from "react";
import Feedback from './Feedback';
import {render, screen, act, waitFor} from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from "@testing-library/user-event";
import { sendToBackend } from "../helper_functions/TalkWithBackend";

jest.mock("../helper_functions/TalkWithBackend");
describe("Feedback_function", () => {
  // Tests that when all required fields are filled out and recaptcha is completed, feedback is successfully submitted and success message is displayed. tags: [happy path]
  test("test_successful_feedback_submission: tests that when all required fields are filled out and ReCAPTCHA is completed, feedback is successfully submitted and success message is displayed.", async () => {
    // Arrange
    render(<Feedback />);
    const firstNameInput = screen.getByPlaceholderText("John");
    const lastNameInput = screen.getByPlaceholderText("Doe");
    const emailInput = screen.getByPlaceholderText("someone@example.com");
    const feedbackInput = screen.getByPlaceholderText("Type your feedback here");
    const recaptcha = screen.getByTestId("recaptcha-feedback");
    const submitButton = screen.getByRole("button", { name: "Submit" });
    const mockSendToBackend = sendToBackend as jest.MockedFunction<typeof sendToBackend>;
    mockSendToBackend.mockResolvedValueOnce({ success: true });

    // Act
    userEvent.type(firstNameInput, "John");
    userEvent.type(lastNameInput, "Doe");
    userEvent.type(emailInput, "someone@example.com");
    userEvent.type(feedbackInput, "This is a test feedback");

    await act(async () => {
      userEvent.click(recaptcha);
    });
    await act(async () => {
      userEvent.click(submitButton);
    });

    // Assert
    expect(screen.getByText("Feedback submitted!")).toBeInTheDocument();
  });

  // Tests that when not all required fields are filled out, error messages are displayed. tags: [edge case]
  test("test_missing_required_fields: tests that when not all required fields are filled out, error messages are displayed.", async () => {
    // Arrange
    render(<Feedback />);
    const firstNameInput = screen.getByPlaceholderText("John");
    const lastNameInput = screen.getByPlaceholderText("Doe");
    const emailInput = screen.getByPlaceholderText("someone@example.com");
    const feedbackInput = screen.getByPlaceholderText("Type your feedback here");
    const recaptcha = screen.getByTestId("recaptcha-feedback");
    const submitButton = screen.getByRole("button", { name: "Submit" });
    const mockSendToBackend = sendToBackend as jest.MockedFunction<typeof sendToBackend>;
    mockSendToBackend.mockResolvedValueOnce({ success: true });

    // Act
    userEvent.type(firstNameInput, "");
    userEvent.type(lastNameInput, "");
    userEvent.type(emailInput, "");
    userEvent.type(feedbackInput, "");
    await act(async () => {
      userEvent.click(recaptcha);
    });
    await act(async () => {
      userEvent.click(submitButton);
    });

    // Assert
    console.log(screen.getByText("First Name cannot be blank"));
    expect(screen.queryByText("First Name cannot be blank")).toBeInTheDocument();
    expect(screen.queryByText("Last Name cannot be blank")).toBeInTheDocument();
    expect(screen.queryByText("Email Cannot be blank")).toBeInTheDocument();
    expect(screen.queryByText("Please enter some feedback")).toBeInTheDocument();
  });
});
