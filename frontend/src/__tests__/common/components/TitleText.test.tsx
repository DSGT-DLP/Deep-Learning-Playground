import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import TitleText from '@/common/components/TitleText';
 
describe('Page', () => {
  it('renders a heading', () => {
    render(<TitleText text="Test Title Text"/>);
 
    const heading = screen.getByRole('heading', { level: 2 });
 
    expect(heading).toBeInTheDocument();
  });
});
