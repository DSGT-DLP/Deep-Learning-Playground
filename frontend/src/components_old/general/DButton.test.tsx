import '@testing-library/jest-dom';
import { render } from '@testing-library/react';
import userEvent from "@testing-library/user-event";
import React from "react";
import DButton from './DButton';

describe("DButton_function", () => {
  // Tests rendering a button with default properties. tags: [happy path]
  it("test_default_button", () => {
    const { getByRole } = render(<DButton />);
    const button = getByRole("button");
    expect(button).toBeInTheDocument();
    expect(button).toHaveClass("btn", "btn-primary");
    expect(button).not.toBeDisabled();
    expect(button).toHaveTextContent("");
  });

  // Tests rendering a button with custom properties. tags: [happy path]
  it("test_custom_button", () => {
    const onClick = jest.fn();
    const style = { backgroundColor: "red" };
    const { getByRole } = render(
      <DButton
        onClick={onClick}
        style={style}
        disabled={true}
        className="custom-class"
      >
        Custom Button
      </DButton>
    );
    const button = getByRole("button");
    expect(button).toBeInTheDocument();
    expect(button).toHaveClass("custom-class");
    expect(button).toHaveStyle("background-color: red");
    expect(button).toBeDisabled();
    userEvent.click(button);
    expect(onClick).toHaveBeenCalledTimes(0);
  });

  // Tests rendering a button without any properties. tags: [edge case]
  it("test_no_props_button", () => {
    const { getByRole } = render(<DButton />);
    const button = getByRole("button");
    expect(button).toBeInTheDocument();
    expect(button).toHaveClass("btn", "btn-primary");
    expect(button).not.toBeDisabled();
    expect(button).toHaveTextContent("");
  });

  // Tests rendering a button with an empty string as children. tags: [edge case]
  it("test_empty_children_button", () => {
    const { getByRole } = render(<DButton> </DButton>);
    const button = getByRole("button");
    expect(button).toBeInTheDocument();
    expect(button).toHaveClass("btn", "btn-primary");
    expect(button).not.toBeDisabled();
    expect(button).toHaveTextContent("");
  });

  // Tests that the button is disabled when the disabled property is true. tags: [general behavior]
  it("test_disabled_button", () => {
    const { getByRole } = render(
      <DButton disabled={true}>Disabled Button</DButton>
    );
    const button = getByRole("button");
    expect(button).toBeInTheDocument();
    expect(button).toHaveClass("btn", "btn-primary");
    expect(button).toBeDisabled();
    expect(button).toHaveTextContent("Disabled Button");
  });

  // Tests the css classes and properties applied to the button. tags: [general behavior]
  it("test_css_button", () => {
    const style = { backgroundColor: "red" };
    const { getByRole } = render(<DButton style={style}>CSS Button</DButton>);
    const button = getByRole("button");
    expect(button).toBeInTheDocument();
    expect(button).toHaveClass("btn", "btn-primary");
    expect(button).not.toBeDisabled();
    expect(button).toHaveTextContent("CSS Button");
    expect(button).toHaveStyle("background-color: red");
  });
});
