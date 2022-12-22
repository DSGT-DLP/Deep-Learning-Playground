import { createGlobalStyle } from 'styled-components';
import './App.css';

const GlobalStyle = createGlobalStyle`
  body {
    background-color: ${props => props.theme.mode === 'dark' ? '#121212' : '#f6f6ff'};  
    color: ${props => props.theme.mode === 'dark' ? '#f6f6ff' : '#121212'};
  }

  .sections {
    background-color: ${props => props.theme.mode === 'dark' ? '#121212' : '#f6f6ff'};
  }

  p {
    color: ${props => props.theme.mode === 'dark' ? '#f6f6ff' : '#121212'};
  }

  #blank-grid p {
    color: ${props => props.theme.mode === 'dark' ? '#f6f6ff' : '#121212'};
  }

  .input-container {
    background-color: ${props => props.theme.mode === 'dark' ? '#27222e' : '#var(--input)'};
  }

  .added-layer-container .input-box {
    background-color: ${props => props.theme.mode === 'dark' ? '#27222e' : 'white'};
  }
  
  #blank-grid {
    border: ${props => props.theme.mode === 'dark' ? '2px solid white' : '2px solid black'};
  }

  .grid-status-display, .grid-status-display-yellow, 
  .grid-status-display-green, .grid-status-display-gray {
    color: ${props => props.theme.mode === 'dark' ? '#f6f6ff' : '#121212'};
    border: ${props => props.theme.mode === 'dark' ? '2px solid white' : '2px solid black'};
  }

`;

export default GlobalStyle;
