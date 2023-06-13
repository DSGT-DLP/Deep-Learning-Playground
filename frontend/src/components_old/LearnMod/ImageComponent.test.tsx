import "@testing-library/jest-dom";
import { render } from "@testing-library/react";
import React from "react";
import ImageComponent from "./ImageComponent";
import { CTypes } from "./LearningModulesContent";

describe("ImageComponent_function", () => {
  // Tests that the imagecomponent renders an image with valid imagedata object. tags: [happy path]
  it("test_image_rendering", () => {
    const imageData = {
      sectionType: "image" as CTypes,
      path: "https://example.com/image.jpg",
      caption: "Example Image",
      attribution: "John Doe",
      licenseLink: "https://example.com/license",
      questionID: undefined,
      content: null,
      answerChoices: null,
      question: null,
      correctAnswer: null,
      minAccuracy: null,
      answer: undefined
    };
    const { getByAltText } = render(<ImageComponent imageData={imageData} />);
    const imageElement = getByAltText(imageData.caption);
    expect(imageElement).toBeInTheDocument();
    expect(imageElement).toHaveAttribute("src", imageData.path);
  });

  // Tests that the imagecomponent sets the correct caption text for the p tag. tags: [general behavior]
  it("test_correct_caption", () => {
    const imageData = {
      sectionType: "image" as CTypes,
      path: "https://example.com/image.jpg",
      caption: "Example Image",
      attribution: "John Doe",
      licenseLink: "https://example.com/license",
      questionID: undefined,
      content: null,
      answerChoices: null,
      question: null,
      correctAnswer: null,
      minAccuracy: null,
      answer: undefined
    };
    const { getByText } = render(<ImageComponent imageData={imageData} />);
    const captionElement = getByText(imageData.caption);
    expect(captionElement).toBeInTheDocument();
  });

  // Tests that the imagecomponent sets the correct href and text for the a tag. tags: [general behavior]
  it("test_correct_attribution", () => {
    const imageData = {
      sectionType: "image" as CTypes,
      path: "https://example.com/image.jpg",
      caption: "Example Image",
      attribution: "John Doe",
      licenseLink: "https://example.com/license",
      questionID: undefined,
      content: null,
      answerChoices: null,
      question: null,
      correctAnswer: null,
      minAccuracy: null,
      answer: undefined
    };
    const { getByText } = render(<ImageComponent imageData={imageData} />);
    const attributionElement = getByText(imageData.attribution);
    expect(attributionElement).toBeInTheDocument();
    expect(attributionElement).toHaveAttribute("href", imageData.licenseLink);
  });
});
