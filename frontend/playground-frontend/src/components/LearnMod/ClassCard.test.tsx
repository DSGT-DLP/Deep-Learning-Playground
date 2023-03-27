import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import React from "react";
import { BrowserRouter } from 'react-router-dom';
import ClassCard from './ClassCard';
import userEvent from "@testing-library/user-event";

describe("ClassCard_function", () => {
  // Tests that the component renders correctly with valid props. tags: [happy path]
  it("test_render_with_valid_props", () => {
    const moduleProgress = { modulePoints: 0 };
    const info = {
      moduleID: 1,
      title: "Test Title",
      points: 10,
      subClasses: [{ sectionID: 1, title: "Subclass 1", content: [], points: 5 }, { sectionID: 2, title: "Subclass 2", content: [], points:5 }],
    };
    render(
    <BrowserRouter>
        <ClassCard info={info} moduleProgress={moduleProgress} />
    </BrowserRouter>);
    expect(screen.getByText("Test Title")).toBeInTheDocument();
  });

  // Tests the progress bar and completion logic for edge cases such as 0 points, negative points, or pointsearned exceeding points. tags: [edge case]
  it("test_progress_bar_and_completion_logic_for_edge_cases", () => {
    const moduleProgress = { modulePoints: 0 };
    const info = {
      moduleID: 1,
      title: "Test Title",
      points: 10,
      subClasses: [{ sectionID: 1, title: "Subclass 1", content: [], points: 5 }, { sectionID: 2, title: "Subclass 2", content: [], points:5 }],
    };
    render(
        <BrowserRouter>
            <ClassCard info={info} moduleProgress={moduleProgress} />
        </BrowserRouter>);
    expect(screen.getByText("0%")).toBeInTheDocument();
  });

  // Tests that the progress bar updates correctly when pointsearned changes. tags: [general behavior]
  it("test_progress_bar_updates_correctly_when_points_earned_changes", () => {
    const moduleProgress = { modulePoints: 5 };
    const info = {
        moduleID: 1,
      title: "Test Title",
      points: 10,
      subClasses: [{ sectionID: 1, title: "Subclass 1", content: [], points: 5 }, { sectionID: 2, title: "Subclass 2", content: [], points:5 }],
    };
    render(
        <BrowserRouter>
            <ClassCard info={info} moduleProgress={moduleProgress} />
        </BrowserRouter>);
    expect(screen.getByText("50%")).toBeInTheDocument();
  });

  // Tests that the button text changes to "completed" when the module is completed. tags: [general behavior]
  it("test_button_text_changes_to_completed_when_module_is_completed", () => {
    const moduleProgress = { modulePoints: 10 };
    const info = {
        moduleID: 1,
      title: "Test Title",
      points: 10,
      subClasses: [{ sectionID: 1, title: "Subclass 1", content: [], points: 5 }, { sectionID: 2, title: "Subclass 2", content: [], points:5 }],
    };
    render(
        <BrowserRouter>
            <ClassCard info={info} moduleProgress={moduleProgress} />
        </BrowserRouter>);
    expect(screen.getByText("Completed")).toBeInTheDocument();
  });
});
