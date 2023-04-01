import "@testing-library/jest-dom";
import { render, screen, } from '@testing-library/react';
import React from "react";
import { COLORS } from "../../constants";
import BackgroundLayout from './BackgroundLayout';
describe("BackgroundLayout_function", () => {
  // Tests that the background color of the layoutbackground object is set correctly. tags: [happy path]
  it("test_sets_correct_background_color", () => {
    render(
      <BackgroundLayout>
        <div data-testid="child"></div>
      </BackgroundLayout>
    );
    const layoutBackground = screen.getByTestId("layoutBackground");
    expect(layoutBackground).toHaveStyle(
      `background-color: ${COLORS.background}`
    );
  });

  // Tests that the padding of the layoutbackground object is set correctly. tags: [happy path]
  it("test_sets_correct_padding", () => {
    render(
      <BackgroundLayout>
        <div data-testid="child"></div>
      </BackgroundLayout>
    );
    const layoutBackground = screen.getByTestId("layoutBackground");
    expect(layoutBackground).toHaveStyle(`padding: 10px`);
  });

  // Tests that the flex properties of the layoutbackground object are set correctly. tags: [happy path]
  it("test_sets_correct_flex_properties", () => {
    const { getByTestId } = render(
      <BackgroundLayout>
        <div data-testid="child"></div>
      </BackgroundLayout>
    );
    const layoutBackground = getByTestId("layoutBackground");
    expect(layoutBackground).toHaveStyle(`display: flex`);
    expect(layoutBackground).toHaveStyle(`flex-direction: row`);
    expect(layoutBackground).toHaveStyle(`flex-wrap: wrap`);
  });

  // Tests that the backgroundlayout component renders without any child components. tags: [edge case]
  it("test_renders_without_children", () => {
    const { queryByTestId } = render(<BackgroundLayout />);
    expect(queryByTestId("child")).toBeNull();
  });

  // Tests that the backgroundlayout component renders with child components. tags: [happy path]
  it("test_renders_with_children", () => {
    const { getByTestId } = render(
      <BackgroundLayout>
        <div data-testid="child"></div>
      </BackgroundLayout>
    );
    expect(getByTestId("child")).toBeInTheDocument();
  });
});
