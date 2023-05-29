import React from 'react';
import { render, screen } from '@testing-library/react';
import About from '../pages/about';

describe('About', () => {
  test('renders heading correctly', () => {
    render(<About />);
    const headingElement = screen.getByTestId('header');
    expect(headingElement).toBeInTheDocument();
    expect(headingElement).toHaveTextContent('Deep Learning Playground');
  });

  test('renders motivation section correctly', () => {
    render(<About />);
    const motivationSection = screen.getByTestId('motivation');
    expect(motivationSection).toBeInTheDocument();
    // Add additional assertions as needed
  });

  // Add more tests for other sections and components
});
