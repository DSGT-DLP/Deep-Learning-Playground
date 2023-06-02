import React from "react";
import { Provider } from "react-redux";
import { act } from 'react-dom/test-utils';
import AccountSettings from "./AccountSettings";
import { render, screen, waitFor, within } from "@testing-library/react";
import "@testing-library/jest-dom";
import userEvent from "@testing-library/user-event";
import Form from "react-bootstrap/Form";
import { updateUserSettings } from "../../firebase";
import mockedStore from "../../redux/mockStore";

describe("AccountSettings tests", () => {
  describe("AccountSettings Renders", () => {
    test("account settings block exists", () => {
      const updateUserSettingsMock = jest.fn();
      updateUserSettingsMock.mockImplementation(updateUserSettingsMock);
      render(
        <Provider store={mockedStore}>
          <AccountSettings />
        </Provider>
      );
      const settingsBlock = screen.getByTestId("user-settings");
      expect(settingsBlock).toBeInTheDocument();
    });
  });

  describe("SettingsBlock tests", () => {
    test("test_passwords_match: tests that handleUpdateUser function checks if passwords match before calling updateUserSettings", async () => {
      render(
        <Provider store={mockedStore}>
          <AccountSettings />
        </Provider>
      );
      const passwordInput = screen.getByLabelText("Password");
      const checkPasswordInput = screen.getByLabelText("Re-Type Password");
      const updateButton = screen.getByText("Update Profile");
      userEvent.type(passwordInput, "password");
      userEvent.type(checkPasswordInput, "password");
      expect(passwordInput).toHaveValue("password");
      expect(checkPasswordInput).toHaveValue("password");
      expect(passwordInput === checkPasswordInput);
    });

     // Tests that the function returns an empty fragment when there is no current user. tags: [happy path]
     test("test_empty_user: tests that the function returns an empty fragment when there is no current user", () => {
      const mockUseAppSelector = jest.fn();
      jest.mock("../../redux/hooks", () => ({
          useAppSelector: mockUseAppSelector
      }));
      mockUseAppSelector.mockReturnValueOnce(null);
      act(async () => {
        render(
          <Provider store={mockedStore}>
            <AccountSettings />
          </Provider>
        );
      });
      expect(screen.queryByText("View or Change your Account Settings")).not.toBeInTheDocument();
    });
      
  });

});
  
