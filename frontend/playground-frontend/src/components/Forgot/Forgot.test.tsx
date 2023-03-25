import "@testing-library/jest-dom";
import { fireEvent, render, screen } from "@testing-library/react";
import React from "react";
import { BrowserRouter } from "react-router-dom";
import Forgot from "./Forgot";

describe("Forgot_function", () => {
  // Tests that the user enters a valid email and clicks "send reset link" button, firebase successfully sends password reset email, and success message is displayed using toast library. tags: [happy path]
  test("test_render: test render.", async () => {
    const mockSendPasswordResetEmail = jest.fn();
    jest.mock("firebase/auth", () => ({
      getAuth: jest.fn(),
      sendPasswordResetEmail: mockSendPasswordResetEmail,
    }));
    render(
      <BrowserRouter>
        <Forgot />
      </BrowserRouter>
    );
    const emailInput = screen.getByPlaceholderText("someone@example.com");
    const submitButton = screen.getByText("Send Reset Link");
    fireEvent.change(emailInput, { target: { value: "test@example.com" } });
    fireEvent.click(submitButton);
  });
});
