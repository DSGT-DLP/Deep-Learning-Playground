import '@testing-library/jest-dom';
import { render } from '@testing-library/react';
import React from "react";
import Spacer from './Spacer';

describe("Spacer_function", () => {
  // Tests that spacer function renders a div with inline styles setting both height and width properties correctly when both height and width properties are provided. tags: [happy path]
  test("test_spacer_with_both_height_and_width_properties", () => {
    const { container } = render(<Spacer height={50} width={100} />);
    expect(container.firstChild).toHaveStyle("height: 50px; width: 100px;");
  });

  // Tests that spacer function renders a div with inline styles setting default height and width properties when no properties are provided. tags: [happy path]
  test("test_spacer_with_no_properties_provided", () => {
    const { container } = render(<Spacer />);
    expect(container.firstChild).toHaveStyle("height: 1px; width: 1px;");
  });

  // Tests that spacer function renders a div with inline styles setting height property correctly when only height property is provided. tags: [happy path]
  test("test_spacer_with_only_height_property", () => {
    const { container } = render(<Spacer height={50} />);
    expect(container.firstChild).toHaveStyle("height: 50px; width: 1px;");
  });

  // Tests that spacer function renders a div with inline styles setting default height and width properties when height or width is provided as 0. tags: [edge case]
  test("test_spacer_with_zero_height_or_width", () => {
    const { container } = render(<Spacer height={0} width={0} />);
    expect(container.firstChild).toHaveStyle("height: 1px; width: 1px;");
  });

  // Tests that spacer function renders a div with inline styles setting width property correctly when only width property is provided. tags: [happy path]
  test("test_spacer_with_only_width_property", () => {
    const { container } = render(<Spacer width={100} />);
    expect(container.firstChild).toHaveStyle("height: 1px; width: 100px;");
  });
});
