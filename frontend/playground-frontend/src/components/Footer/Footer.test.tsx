import React from "react";
import Footer from './Footer';
import {render, screen} from '@testing-library/react';
import '@testing-library/jest-dom';
import { URLs } from "../../constants";
import userEvent from "@testing-library/user-event";

describe("Footer component", () => {
    // Tests that social media icons are rendered with correct colors and sizes, and clicking on them directs the user to the correct url. tags: [happy path, edge case]
    test("test_social_media_icons: tests that social media icons are rendered with correct colors and sizes, and clicking on them directs the user to the correct URL", () => {
        render(<Footer />);
        const linkedinIcon = screen.getByTestId("linkedin-icon");
        const youtubeIcon = screen.getByTestId("youtube-icon");
        const instagramIcon = screen.getByTestId("instagram-icon");
        const githubIcon = screen.getByTestId("github-icon");

        expect(linkedinIcon).toHaveAttribute("href", URLs.linkedin);

        expect(youtubeIcon).toHaveAttribute("href", URLs.youtube);

        expect(instagramIcon).toHaveAttribute("href", URLs.instagram);

        expect(githubIcon).toHaveAttribute("href", URLs.github);
    });

    // Tests that the social media icons are rendered in the correct order. tags: [happy path]
    test("test_icon_order: tests that the social media icons are rendered in the correct order", () => {
        render(<Footer />);
        const socialsDiv = screen.getByTestId("footer-socials");
        const socialIcons = socialsDiv.children;

        expect(socialIcons[0]).toHaveClass("foot-element");
        expect(socialIcons[0]).toContainElement(screen.getByTestId("linkedin-icon"));

        expect(socialIcons[1]).toHaveClass("foot-element");
        expect(socialIcons[1]).toContainElement(screen.getByTestId("youtube-icon"));

        expect(socialIcons[2]).toHaveClass("foot-element");
        expect(socialIcons[2]).toContainElement(screen.getByTestId("instagram-icon"));

        expect(socialIcons[3]).toHaveClass("foot-element");
        expect(socialIcons[3]).toContainElement(screen.getByTestId("github-icon"));
    });

    // Tests that the copyright notice displays the correct text, including the current year. tags: [happy path]
    test("test_copyright_notice: tests that the copyright notice displays the correct text, including the current year", () => {
        render(<Footer />);
        const footerNameDiv = screen.getByTestId("footer-name");
        const currentYear = new Date().getFullYear();

        expect(footerNameDiv).toContainHTML(`Deep Learning Playground © ${currentYear}`);
    });

    // Tests for keyboard navigation and screen reader compatibility. tags: [edge case]
    test("test_accessibility: tests for keyboard navigation and screen reader compatibility", () => {
        render(<Footer />);
        const linkedinIcon = screen.getByTestId("linkedin-icon");
        const youtubeIcon = screen.getByTestId("youtube-icon");
        const instagramIcon = screen.getByTestId("instagram-icon");
        const githubIcon = screen.getByTestId("github-icon");

        expect(linkedinIcon).toHaveAccessibleDescription("Link to LinkedIn profile");
        expect(youtubeIcon).toHaveAccessibleDescription("Link to YouTube channel");
        expect(instagramIcon).toHaveAccessibleDescription("Link to Instagram profile");
        expect(githubIcon).toHaveAccessibleDescription("Link to GitHub repository");

        expect(linkedinIcon).toBeEnabled();
        expect(youtubeIcon).toBeEnabled();
        expect(instagramIcon).toBeEnabled();
        expect(githubIcon).toBeEnabled();

        userEvent.tab();
        userEvent.keyboard("{enter}");
        expect(document.activeElement).toEqual(linkedinIcon);
        expect(document.activeElement?.getAttribute("href")).toEqual(URLs.linkedin);
        
        userEvent.tab();
        userEvent.keyboard("{enter}");
        expect(document.activeElement).toEqual(youtubeIcon);
        expect(document.activeElement?.getAttribute("href")).toEqual(URLs.youtube);
        
        userEvent.tab();
        userEvent.keyboard("{enter}");
        expect(document.activeElement).toEqual(instagramIcon);
        expect(document.activeElement?.getAttribute("href")).toEqual(URLs.instagram);

        userEvent.tab();
        userEvent.keyboard("{enter}");
        expect(document.activeElement).toEqual(githubIcon);
        expect(document.activeElement?.getAttribute("href")).toEqual(URLs.github);

    });
});
