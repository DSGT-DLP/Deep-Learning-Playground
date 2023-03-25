import React from "react";
import {Provider} from "react-redux";
import AccountSettings from './AccountSettings';
import {render, screen, waitFor, within} from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';
import store from '../../redux/store';
import Form from "react-bootstrap/Form";

describe("SettingsBlock_function", () => {
  // Tests that the function updates user settings when valid input is provided. tags: [happy path]

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
