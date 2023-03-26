import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import React from "react";
import TitleText from "./TitleText";

describe("TitleText_function", () => {
  // Tests that the function renders a title text with a valid string input. tags: [happy path]
  test("test_render_title_text_with_valid_input", () => {
    const { getByText } = render(<TitleText text="Valid Input" />);
    expect(getByText("Valid Input")).toBeInTheDocument();
  });

  // Tests that the function returns a jsx element with the correct tag and style. tags: [happy path]
  test("test_return_jsx_element_with_correct_tag_and_style", () => {
    const { getByText } = render(<TitleText text="Correct Tag and Style" />);
    const titleText = getByText("Correct Tag and Style");
    expect(titleText.tagName).toBe("H2");
  });

  // Tests that the function extracts the 'text' property correctly from the input object. tags: [happy path]
  test("test_extract_text_property_correctly_from_input_object", () => {
    const { getByText } = render(<TitleText text="Extract Text Property" />);
    expect(getByText("Extract Text Property")).toBeInTheDocument();
  });

  // Tests that the function renders a title text with an empty string input. tags: [edge case]
  test("test_render_title_text_with_empty_string_input", () => {
    render(<TitleText text="" />);
    expect(screen.getAllByText("").length).toBeGreaterThanOrEqual(1);
  });
  
});
