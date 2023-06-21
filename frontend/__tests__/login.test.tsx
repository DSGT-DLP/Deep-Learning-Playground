import React from 'react';
import { render, screen } from '@testing-library/react';
import Login from '../src/pages/login';
import '@testing-library/jest-dom';
 
describe('Login', () => {
  it('renders a heading', () => {
    render(<Login />);
 
    const heading = screen.getByRole('heading', {
      name: /No-code Solution/i,
    });
 
    expect(heading).toBeInTheDocument();
  });
});