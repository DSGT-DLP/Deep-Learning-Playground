import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom";
import ModulesSideBar from "./ModulesSideBar";
import getContent from './LearningModulesContent';
import { sendToBackend } from "../helper_functions/TalkWithBackend";
import { CTypes } from "./LearningModulesContent";
import {BrowserRouter} from 'react-router-dom';
describe("ModulesSideBar_function", () => {
  // Tests that the sidebar is rendered with the correct list of modules and subsections. tags: [happy path]
  it("test_sidebar_is_rendered", () => {
    render(<BrowserRouter><ModulesSideBar /></BrowserRouter>);
    const modules = getContent.modules;
    modules.forEach((module) => {
      expect(screen.getByText(module.title)).toBeInTheDocument();
      module.subClasses.forEach((subsection) => {
        expect(screen.getByText(subsection.title)).toBeInTheDocument();
      });
    });
  });

  // Tests for accessibility and responsiveness of the sidebar. tags: [edge case]
  it("test_accessibility_and_responsiveness", () => {
    render(<BrowserRouter><ModulesSideBar /></BrowserRouter>);
    waitFor(() => {
        const sideBar = screen.getByRole("complementary");
        expect(sideBar).toHaveAccessibleName("Modules Sidebar");
        expect(sideBar).toHaveStyle({ display: "block" });
        expect(sideBar).toHaveStyle({ width: "20%" });
        expect(sideBar).toHaveStyle({ backgroundColor: "#333" });
        expect(sideBar).toHaveStyle({ color: "white" });
        expect(sideBar).toHaveStyle({ padding: "1rem" });
        expect(sideBar).toHaveStyle({ position: "fixed" });
        expect(sideBar).toHaveStyle({ top: "0" });
        expect(sideBar).toHaveStyle({ left: "0" });
        expect(sideBar).toHaveStyle({ height: "100%" });
    });
  });

  // Tests that the sidebar is initially closed and the toggle button displays the correct text. tags: [edge case]
  it("test_sidebar_initially_closed", () => {
    render(<BrowserRouter><ModulesSideBar /></BrowserRouter>);
    waitFor(() => {
        const sideBar = screen.getByRole("complementary");
        const toggleButton = screen.getByRole("button");
        expect(sideBar).toHaveStyle({display: "none"});
        expect(toggleButton).toHaveTextContent(">>");
    });
  });

  // Tests that the state variables are updated correctly on toggle. tags: [general behavior]
  it("test_sidebar_toggle_updates_state", () => {
    render(<BrowserRouter><ModulesSideBar /></BrowserRouter>);
    waitFor(() => {
        let toggleButton = screen.getByRole("button");
        userEvent.click(toggleButton);
        toggleButton = screen.getByRole("button");
        userEvent.click(toggleButton);
        expect(toggleButton).toHaveTextContent("<<");
        expect(screen.getByRole("complementary")).toHaveStyle({ display: "block" });
        userEvent.click(toggleButton);
        expect(toggleButton).toHaveTextContent(">>");
        expect(screen.getByRole("complementary")).toHaveStyle({ display: "none" });
    });
    
  });

  // Tests that inline styling is applied correctly to rendered elements. tags: [general behavior]
  it("test_inline_styling", () => {
    render(<BrowserRouter><ModulesSideBar /></BrowserRouter>);
    waitFor(() => {
        const sideBar = screen.getByRole("complementary");
        const toggleButton = screen.getByRole("button");
        expect(sideBar).toHaveStyle({ backgroundColor: "#333" });
        expect(sideBar).toHaveStyle({ color: "white" });
        expect(sideBar).toHaveStyle({ padding: "1rem" });
        expect(sideBar).toHaveStyle({ position: "fixed" });
        expect(sideBar).toHaveStyle({ top: "0" });
        expect(sideBar).toHaveStyle({ left: "0" });
        expect(sideBar).toHaveStyle({ height: "100%" });
        expect(toggleButton).toHaveStyle({ backgroundColor: "#333" });
        expect(toggleButton).toHaveStyle({ color: "white" });
        expect(toggleButton).toHaveStyle({ border: "none" });
        expect(toggleButton).toHaveStyle({ padding: "0.5rem" });
    });
    
  });
});
