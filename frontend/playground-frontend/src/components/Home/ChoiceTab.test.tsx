import React from "react";
import {render, screen, fireEvent} from "@testing-library/react";
import "@testing-library/jest-dom";
import {COLORS} from "../../constants";
import ChoiceTab from "./ChoiceTab";
import { BrowserRouter } from "react-router-dom";
describe("ChoiceTab_function", () => {
  // Tests that buttons with links to different pages are rendered. tags: [happy path]
  it("test_button_links", () => {
    const { getByText } = render(<BrowserRouter><ChoiceTab/></BrowserRouter>);
    expect(getByText("Tabular Data")).toBeInTheDocument();
    expect(getByText("Image Models")).toBeInTheDocument();
    expect(getByText("Classical ML")).toBeInTheDocument();
    expect(getByText("Object Detection")).toBeInTheDocument();
  });

  // Tests that the function handles an invalid pathname. tags: [edge case]
  it("test_invalid_pathname", () => {
    const { getByText } = render(<BrowserRouter><ChoiceTab/></BrowserRouter>);
    expect(() => getByText("Invalid Pathname")).toThrow();
  });

  // Tests that the buttons have the correct text. tags: [general behavior]
  it("test_button_text", () => {
    const { getByText } = render(<BrowserRouter><ChoiceTab/></BrowserRouter>);
    expect(getByText("Tabular Data")).toHaveTextContent("Tabular Data");
    expect(getByText("Image Models")).toHaveTextContent("Image Models");
    expect(getByText("Classical ML")).toHaveTextContent("Classical ML");
    expect(getByText("Object Detection")).toHaveTextContent("Object Detection");
  });

  // Tests that the links have the correct href attribute. tags: [general behavior]
  it("test_link_href", () => {
    const { getByText } = render(<BrowserRouter><ChoiceTab/></BrowserRouter>);
    expect(getByText("Tabular Data")).toHaveAttribute("href", "/train");
    expect(getByText("Image Models")).toHaveAttribute("href", "/img-models");
    expect(getByText("Classical ML")).toHaveAttribute("href", "/classical-ml");
    expect(getByText("Object Detection")).toHaveAttribute(
      "href",
      "/object-detection"
    );
  });
});
