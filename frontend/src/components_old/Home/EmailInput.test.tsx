import "@testing-library/jest-dom";
import { fireEvent, render, waitFor } from "@testing-library/react";
import React from "react";
import EmailInput from "./EmailInput";
describe("EmailInput_function", () => {
  // Tests that the email state variable is updated on change. tags: [happy path]
  it("test_email_input_updates_state", () => {
    const setEmail = jest.fn();
    const { getByPlaceholderText } = render(<EmailInput setEmail={setEmail} />);
    const input = getByPlaceholderText("someone@example.com");
    fireEvent.change(input, { target: { value: "test@test.com" } });
    expect(setEmail).toHaveBeenCalledWith("test@test.com");
  });

  // Tests that the emailnotvalid state variable is updated on blur. tags: [happy path]
  it("test_email_input_validates_email", () => {
    const setEmail = jest.fn();
    const { getByPlaceholderText, getByText } = render(
      <EmailInput setEmail={setEmail} />
    );
    const input = getByPlaceholderText("someone@example.com");
    fireEvent.change(input, { target: { value: "test@test.com" } });
    fireEvent.blur(input);
    waitFor(() => {
      expect(getByText("Please enter a valid email")).toBeInTheDocument();
    });
  });

  // Tests that the component behaves correctly when an empty email is entered. tags: [happy path]
  it("test_email_input_empty_email", () => {
    const setEmail = jest.fn();
    const { getByPlaceholderText, queryByText } = render(
      <EmailInput setEmail={setEmail} />
    );
    const input = getByPlaceholderText("someone@example.com");
    fireEvent.change(input, { target: { value: "" } });
    fireEvent.blur(input);
    expect(queryByText("Please enter a valid email")).toBeNull();
  });

  // Tests that the component behaves correctly when an email with invalid format is entered. tags: [edge case]
  it("test_email_input_invalid_format", () => {
    const setEmail = jest.fn();
    const { getByPlaceholderText, getByText } = render(
      <EmailInput setEmail={setEmail} />
    );
    const input = getByPlaceholderText("someone@example.com");
    fireEvent.change(input, { target: { value: "test" } });
    fireEvent.blur(input);
    waitFor(() => {
      expect(getByText("Please enter a valid email")).toBeInTheDocument();
    });
  });

  // Tests that the component behaves correctly when an email with more than 255 characters is entered. tags: [edge case]
  it("test_email_input_long_email", () => {
    const setEmail = jest.fn();
    const { getByPlaceholderText, getByText } = render(
      <EmailInput setEmail={setEmail} />
    );
    const input = getByPlaceholderText("someone@example.com");
    const longEmail = "a".repeat(256) + "@test.com";
    fireEvent.change(input, { target: { value: longEmail } });
    fireEvent.blur(input);
    waitFor(() => {
      expect(getByText("Please enter a valid email")).toBeInTheDocument();
    });
  });
});
