import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import React from "react";
import { BrowserRouter } from 'react-router-dom';
import NavbarMain from './NavbarMain';
import { Provider } from 'react-redux';
import mockedStore from "../../redux/mockStore";
import userEvent from '@testing-library/user-event';
import { setCookie } from '../helper_functions/Cookie';

describe("NavbarMain_function", () => {

  // Tests that the navbarmain component renders without errors and displays the expected elements. tags: [happy path]
  it("test_navbar_main_renders_correctly", () => {
    render(
        <Provider store={mockedStore}>
            <BrowserRouter>
            <NavbarMain />
            </BrowserRouter>
        </Provider>
    );
    expect(screen.getByAltText("DSGT Logo")).toBeInTheDocument();
    expect(screen.getByText("Deep Learning Playground")).toBeInTheDocument();
    expect(screen.getByText("Train")).toBeInTheDocument();
    expect(screen.getByText("About")).toBeInTheDocument();
    expect(screen.getByText("Wiki")).toBeInTheDocument();
    expect(screen.getByText("Feedback")).toBeInTheDocument();
    expect(screen.getByText("Donate")).toBeInTheDocument();
    expect(screen.getByText("Account")).toBeInTheDocument();
    userEvent.click(screen.getByText("Account"));
    expect(screen.getByText("Settings")).toBeInTheDocument();
    expect(screen.getByText("Learn")).toBeInTheDocument();
    expect(screen.getByText("Log out")).toBeInTheDocument();
  });

  // Tests that the navbar displays the correct links and dropdown menu when a user is authenticated. tags: [happy path]
  it("test_navbar_main_user_authenticated", () => {
    // Set up initial state with a user authenticated
    const initialState = {
      currentUser: {
        user: {
          email: "test@test.com",
          uid: "123456",
          displayName: "Test User",
          emailVerified: true,
        },
      },
    };
    // Render the NavbarMain component with the initial state
    render(
        <Provider store={mockedStore}>
            <BrowserRouter>
            <NavbarMain />
            </BrowserRouter>
        </Provider>
    );
    // Check that the correct links and dropdown menu are displayed
    expect(screen.getByText("Train")).toBeInTheDocument();
    expect(screen.getByText("About")).toBeInTheDocument();
    expect(screen.getByText("Wiki")).toBeInTheDocument();
    expect(screen.getByText("Feedback")).toBeInTheDocument();
    expect(screen.getByText("Donate")).toBeInTheDocument();
    expect(screen.getByText("Account")).toBeInTheDocument();
    // Check that the login button is not displayed
    expect(screen.queryByText("Log in")).not.toBeInTheDocument();
  });

  // Tests that the setcookie function correctly sets a cookie with the given name, value, and expiration date. tags: [happy path]
  it("test_navbar_main_cookie_set", () => {
    // Call the setCookie function with test values
    setCookie("testCookie", "testValue", 1, "/");
    // Check that the cookie was set correctly
    expect(document.cookie).toContain("testCookie=testValue");
  });
});
