import React from "react";
import {fireEvent, screen, render} from "@testing-library/react";
import RectContainer from "./RectContainer";
import "@testing-library/jest-dom";
import {act} from "react-dom/test-utils";

describe("RectContainer_function", () => {
  // Tests rendering a div container with all valid props. tags: [happy path]
  it("test_valid_props", () => {
    const props = {
      children: "Test",
      style: { backgroundColor: "red" },
      className: "test-class",
      containerRef: React.createRef<HTMLDivElement>(),
      dataTestId: "test-id",
    };
    const { getByTestId } = render(<RectContainer {...props} />);
    const container = getByTestId("test-id");
    expect(container).toBeInTheDocument();
    expect(container).toHaveStyle("background-color: red");
    expect(container).toHaveClass("test-class");
    expect(container.textContent).toBe("Test");
  });

  // Tests rendering a div container with different types of child components (e.g. strings, numbers, other react components). tags: [edge case]
  it("test_child_components", () => {
    const props = {
      children: ["Test", <span key="1">Span</span>, 123],
      style: { backgroundColor: "red" },
      className: "test-class",
      containerRef: React.createRef<HTMLDivElement>(),
      dataTestId: "test-id",
    };
    const { getByTestId } = render(<RectContainer {...props} />);
    const container = getByTestId("test-id");
    expect(container).toBeInTheDocument();
    expect(container).toHaveStyle("background-color: red");
    expect(container).toHaveClass("test-class");
    expect(container.textContent).toBe("TestSpan123");
  });

  // Tests rendering a div container with different types of data-testid props (e.g. strings, numbers, other react components). tags: [edge case]
  it("test_data_test_id", () => {
    const props = {
      children: "Test",
      style: { backgroundColor: "red" },
      className: "test-class",
      containerRef: React.createRef<HTMLDivElement>(),
      dataTestId: "123",
    };
    const { getByTestId } = render(<RectContainer {...props} />);
    const container = getByTestId("123");
    expect(container).toBeInTheDocument();
    expect(container).toHaveStyle("background-color: red");
    expect(container).toHaveClass("test-class");
    expect(container.textContent).toBe("Test");
  });

});
