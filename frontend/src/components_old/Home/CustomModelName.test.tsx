import "@testing-library/jest-dom";
import { fireEvent, render, waitFor } from "@testing-library/react";
import React from "react";
import CustomModelName from "./CustomModelName";

describe("CustomModelName_function", () => {
  // Tests that the function updates the state with the new value when the user leaves the input field. tags: [happy path]
  it("test_updates_state_on_blur", () => {
    const setCustomModelName = jest.fn();
    const { getByPlaceholderText } = render(
      <CustomModelName
        customModelName=""
        setCustomModelName={setCustomModelName}
      />
    );
    const input = getByPlaceholderText("Give a custom model name");
    fireEvent.blur(input, { target: { value: "new model name" } });
    expect(setCustomModelName).toHaveBeenCalledWith("new model name");
  });

  // Tests that the function accepts a valid custom model name. tags: [happy path, edge case]
  it("test_valid_custom_model_name", () => {
    const setCustomModelName = jest.fn();
    const { getByPlaceholderText } = render(
      <CustomModelName
        customModelName=""
        setCustomModelName={setCustomModelName}
      />
    );
    const input = getByPlaceholderText("Give a custom model name");
    fireEvent.blur(input, { target: { value: "valid model name" } });
    expect(setCustomModelName).toHaveBeenCalledWith("valid model name");
  });

  // Tests that the function handles an empty custom model name. tags: [edge case]
  it("test_empty_custom_model_name", () => {
    const setCustomModelName = jest.fn();
    const { getByPlaceholderText } = render(
      <CustomModelName
        customModelName=""
        setCustomModelName={setCustomModelName}
      />
    );
    const input = getByPlaceholderText("Give a custom model name");
    fireEvent.blur(input, { target: { value: "" } });
    expect(setCustomModelName).toHaveBeenCalledWith("");
  });

  // Tests that the function handles a custom model name that exceeds the maximum length. tags: [edge case]
  it("test_custom_model_name_exceeds_max_length", () => {
    const setCustomModelName = jest.fn();
    const { getByPlaceholderText } = render(
      <CustomModelName
        customModelName=""
        setCustomModelName={setCustomModelName}
      />
    );
    const input = getByPlaceholderText("Give a custom model name");
    fireEvent.blur(input, { target: { value: "a".repeat(256) } });
    waitFor(() => {expect(setCustomModelName).toHaveBeenCalledWith("a".repeat(255));});
  });

  // Tests that the function handles errors gracefully. tags: [general behavior]
  it("test_handles_errors_gracefully", () => {
    const setCustomModelName = jest.fn();
    const { getByPlaceholderText } = render(
      <CustomModelName
        customModelName=""
        setCustomModelName={setCustomModelName}
      />
    );
    const input = getByPlaceholderText("Give a custom model name");
    fireEvent.blur(input, { target: { value: null } });
    waitFor(() => {expect(setCustomModelName).not.toHaveBeenCalled();});
  });
});
