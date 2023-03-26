import '@testing-library/jest-dom';
import { render } from '@testing-library/react';
import userEvent from "@testing-library/user-event";
import React from "react";
import ImageFileUpload from './ImageFileUpload';

describe("ImageFileUpload_function", () => {
  // Tests that when a user selects a jpg file, the uploadfile state is set to the selected file. tags: [happy path]
  it("test_file_upload_success", () => {
    const setUploadFile = jest.fn();
    const { getByLabelText } = render(
      <ImageFileUpload uploadFile={null} setUploadFile={setUploadFile} />
    );
    const file = new File(["(⌐□_□)"], "test.jpg", { type: "image/jpeg" });
    const input = getByLabelText("Choose jpg file");
    userEvent.upload(input, file);
    expect(setUploadFile).toHaveBeenCalledWith(file);
  });

  // Tests that when a user does not select a file, the uploadfile state remains null. tags: [edge case]
  it("test_no_file_selected", () => {
    const setUploadFile = jest.fn();
    render(<ImageFileUpload uploadFile={null} setUploadFile={setUploadFile} />);
    expect(setUploadFile).not.toHaveBeenCalled();
  });

});
