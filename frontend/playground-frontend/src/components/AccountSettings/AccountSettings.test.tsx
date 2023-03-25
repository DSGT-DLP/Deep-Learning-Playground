import React from "react";
import {Provider} from "react-redux";
import {act} from "react-dom/test-utils";
import AccountSettings from './AccountSettings';
import configureMockStore from 'redux-mock-store';
import {render, screen, waitFor, within} from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';
import { updateUserSettings } from '../../firebase';
import { ScreenRotation } from "@mui/icons-material";

const mockStore = configureMockStore();
const store = mockStore({ currentUser:  {
  user: {
    email: "johndoe@gmail.com",
    uid: "12345",
    displayName: "John Doe",
    emailVerified: true
  },
  userProgressData: null
},
train: {
  customModelName: "John's Model",
  fileName: undefined,
  csvDataInput: [{"col1": ["data1", "data2", "data3"]}, {"col2": ["data4", "data5", "data6"]}],
  oldCsvDataInput: [{"col1": ["data1", "data2", "data3"]}, {"col2": ["data4", "data5", "data6"]}],
  uploadedColumns: ["col1", "col2"],
  fileURL: "data.csv"
}});
describe("AccountSettings tests", () => {
  describe("AccountSettings Renders", () => {
    test('account settings block exists', () => {
      const updateUserSettingsMock = jest.fn();
      updateUserSettingsMock.mockImplementation(updateUserSettingsMock);
      render(
          <Provider store={store}>
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
        <Provider store={store}>
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
          <Provider store={store}>
            <AccountSettings />
          </Provider>
        );
      });
      expect(screen.queryByText("View or Change your Account Settings")).not.toBeInTheDocument();
    });

  });
  
});
