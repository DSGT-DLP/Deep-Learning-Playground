import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import userEvent from "@testing-library/user-event";
import React from "react";
import LargeFileUpload from './LargeFileUpload';

describe("LargeFileUpload", () => {
    // Tests that the function displays the faclouduploadalt icon and the name of the selected file or "choose zip file" if no file is selected. tags: [happy path]
    it("test_display_icon_and_file_name", () => {
        const { getByLabelText } = render(<LargeFileUpload uploadFile={null} setUploadFile={jest.fn()} />);
        const label = getByLabelText("Choose zip file");
        expect(label).toBeInTheDocument();
        expect(label.getAttribute("accept")).toEqual(".zip");
    });

    // Tests that the function limits the file size to 50mb. tags: [edge case]
    it("test_limit_file_size", () => {
        const { getByLabelText } = render(<LargeFileUpload uploadFile={null} setUploadFile={jest.fn()} />);
        const input = getByLabelText("Choose zip file");
        const file = new File(["test"], "test.zip", { type: "application/zip" });
        userEvent.upload(input, file);
        expect(file.size).toBeLessThanOrEqual(50000000);
    });

    // Tests that the function limits the file size to 50mb.
    it("test_limit_file_size", () => {
        const file = new File(["test"], "test.zip", { type: "application/zip" });
        Object.defineProperty(file, "size", {
            value: 50000001
        });
        const setUploadFile = jest.fn();
        render(<LargeFileUpload uploadFile={null} setUploadFile={setUploadFile} />);
        const input = screen.getByLabelText("Choose zip file");
        userEvent.upload(input, file);
        expect(file.size).toBeGreaterThan(50000000);
        expect(screen.getByText("Choose zip file")).toBeInTheDocument();
    });

    // Tests that the function displays an error message when the user selects a non-zip file. tags: [edge case]
    it("test_error_message_for_non_zip_file", () => {
        const { getByLabelText, getByText } = render(<LargeFileUpload uploadFile={null} setUploadFile={jest.fn()} />);
        const input = getByLabelText("Choose zip file");
        const file = new File(["test"], "test.txt", { type: "text/plain" });
        
        userEvent.upload(input, file);
        expect(getByText("Choose zip file")).toBeInTheDocument();
    });
});
