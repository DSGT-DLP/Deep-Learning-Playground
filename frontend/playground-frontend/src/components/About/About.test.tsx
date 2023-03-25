import React from "react";
import About from './About';
import {render, screen} from '@testing-library/react';
import '@testing-library/jest-dom';
import { URLs } from "../../constants";

describe("About_function", () => {
  // Tests that the heading section is rendered correctly. tags: [happy path]
  test("test_render_heading: tests that the heading section is rendered correctly", () => {
    render(<About />);
    const title = screen.getByTestId("header");
    expect(title.getElementsByClassName("headers")).toHaveLength(1);
    expect(title.getElementsByClassName("headers")[0]).toHaveTextContent("Deep Learning Playground");
  });

  // Tests that the motivation section is rendered correctly. tags: [happy path]
  test("test_render_motivation_section: tests that the motivation section is rendered correctly", () => {
    render(<About />);
    const motivation = screen.getByTestId("motivation");
    expect(motivation.getElementsByTagName('h2')[0]).toHaveTextContent("Motivation");
  });

  // Tests that the technologies used section is rendered correctly. tags: [happy path]
  test("test_render_technologies_used_section: tests that the technologies used section is rendered correctly", () => {
    render(<About />);
    const tech = screen.getByTestId("tech");
    const techContent = screen.getAllByTestId("tech-row-content");
    expect(tech.getElementsByTagName('h2')[0]).toHaveTextContent("Technologies Used");
    console.log(tech.getElementsByClassName("tech-rows"));
    expect(techContent).toHaveLength(6);
  });

  // Tests that the installation instructions section is rendered correctly. tags: [happy path]
  test("test_render_install_instructions: tests that the installation instructions section is rendered correctly", () => {
    render(<About />);
    const installation = screen.getByTestId("installation");
    expect(installation.getElementsByTagName("h2")).toHaveLength(1);
    expect(installation.getElementsByTagName("h2")[0]).toHaveTextContent("Installation");
    expect(screen.getByRole('link')).toHaveAttribute('href', URLs.github);
  });

  // Tests that the user instructions section is rendered correctly. tags: [happy path]
  test("test_render_user_instructions: tests that the user instructions section is rendered correctly", () => {
    render(<About />);
    const user = screen.getByTestId("user");
    expect(user.getElementsByTagName("h2")).toHaveLength(1);
    expect(user.getElementsByTagName("h2")[0]).toHaveTextContent("User");
    expect(user.getElementsByTagName("p").length).toEqual(2);
  });

  // Tests that the developer instructions section is rendered correctly. tags: [happy path]
  test("test_render_developer_instructions: tests that the developer instructions section is rendered correctly", () => {
    render(<About />);
    const developer = screen.getByTestId("dev");
    expect(developer.getElementsByTagName("h2")[0]).toHaveTextContent("Development");
    expect(developer.getElementsByTagName("p").length).toEqual(2);
    const paragraphs = developer.getElementsByTagName("p");
    expect(paragraphs).toHaveLength(2);
    const devGreeting = screen.getByTestId("developer-greeting");
    expect(devGreeting.textContent).toContain("Want to contribute?");
  });
});
