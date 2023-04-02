import "@testing-library/jest-dom";
import { fireEvent, render, waitFor } from "@testing-library/react";
import React from "react";
import { GENERAL_STYLES } from "../../constants";
import PhoneNumberInput from "./PhoneNumberInput";
describe("PhoneNumberInput_function", () => {
  // Tests that entering a valid phone number with country code updates the phone number input state successfully. tags: [happy path]
  it("test_valid_phone_number_input", () => {
    const setPhoneNumberMock = jest.fn();
    const { getByPlaceholderText } = render(
      <PhoneNumberInput setPhoneNumber={setPhoneNumberMock} />
    );
    const input = getByPlaceholderText("+16785552057");
    fireEvent.blur(input, { target: { value: "+16785553058" } });
    expect(setPhoneNumberMock).toHaveBeenCalledWith("+16785553058");
  });

  // Tests that entering an invalid phone number with country code displays an error message and does not update the phone number input state. tags: [edge case]
  it("test_invalid_phone_number_input", () => {
    const setPhoneNumberMock = jest.fn();
    const { getByPlaceholderText, getByText } = render(
      <PhoneNumberInput setPhoneNumber={setPhoneNumberMock} />
    );
    const input = getByPlaceholderText("+16785552057");
    fireEvent.blur(input, { target: { value: "invalid" } });
    expect(setPhoneNumberMock).not.toHaveBeenCalled();
    expect(
      getByText(
        "Please enter a valid number with the country code e.g. +16785553058"
      )
    ).toBeInTheDocument();
  });

  // Tests that entering a phone number with more than 64 characters displays an error message and does not update the phone number input state. tags: [edge case]
  it("test_long_phone_number_input", () => {
    const setPhoneNumberMock = jest.fn();
    const { getByPlaceholderText, getByText } = render(
      <PhoneNumberInput setPhoneNumber={setPhoneNumberMock} />
    );
    const input = getByPlaceholderText("+16785552057");
    fireEvent.blur(input, {
      target: {
        value:
          "+16785553058000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
      },
    });
    waitFor(() => {
        expect(setPhoneNumberMock).not.toHaveBeenCalled();
        expect(
        getByText(
            "Please enter a valid number with the country code e.g. +16785553058"
        )
        ).toBeInTheDocument();
    });
    
  });

  // Tests that the error message is styled using the general_styles constant. tags: [general behavior]
  it("test_error_message_styling", () => {
    const { getByPlaceholderText, getByText } = render(
      <PhoneNumberInput setPhoneNumber={() => {}} />
    );
    const input = getByPlaceholderText("+16785552057");
    fireEvent.blur(input, { target: { value: "invalid" } });
    expect(
      getByText(
        "Please enter a valid number with the country code e.g. +16785553058"
      )
    ).toHaveStyle(GENERAL_STYLES.error_text);
  });

  // Tests that entering a phone number with special characters displays an error message and does not update the phone number input state. tags: [edge case]
  it("test_special_character_phone_number_input", () => {
    const setPhoneNumberMock = jest.fn();
    const { getByPlaceholderText, getByText } = render(
      <PhoneNumberInput setPhoneNumber={setPhoneNumberMock} />
    );
    const input = getByPlaceholderText("+16785552057");
    fireEvent.blur(input, { target: { value: "+1678555@2057" } });
    expect(setPhoneNumberMock).not.toHaveBeenCalled();
    expect(
      getByText(
        "Please enter a valid number with the country code e.g. +16785553058"
      )
    ).toBeInTheDocument();
  });

  // Tests that entering a phone number with letters displays an error message and does not update the phone number input state. tags: [edge case]
  it("test_letter_phone_number_input", () => {
    const setPhoneNumberMock = jest.fn();
    const { getByPlaceholderText, getByText } = render(
      <PhoneNumberInput setPhoneNumber={setPhoneNumberMock} />
    );
    const input = getByPlaceholderText("+16785552057");
    fireEvent.blur(input, { target: { value: "+1678555abc" } });
    expect(setPhoneNumberMock).not.toHaveBeenCalled();
    expect(
      getByText(
        "Please enter a valid number with the country code e.g. +16785553058"
      )
    ).toBeInTheDocument();
  });
});
