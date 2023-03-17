import { createGlobalStyle } from "styled-components";

const GlobalStyle = createGlobalStyle`
  body {
    background-color: ${(props) =>
      props.theme.mode === "dark" ? "#121212" : "#f6f6ff"};  
    color: ${(props) => (props.theme.mode === "dark" ? "#f6f6ff" : "#121212")};
  }

  .sections {
    background-color: ${(props) =>
      props.theme.mode === "dark" ? "#121212" : "#f6f6ff"};
  }

  p, .inputText {
    color: ${(props) => (props.theme.mode === "dark" ? "#f6f6ff" : "#121212")};
  }

  #blank-grid p {
    color: ${(props) => (props.theme.mode === "dark" ? "#f6f6ff" : "#121212")};
  }

  .input-container {
    background-color: ${(props) =>
      props.theme.mode === "dark" ? "#5A5A5A" : "#var(--input)"};
  }

  .added-layer-container .input-box {
    background-color: ${(props) =>
      props.theme.mode === "dark" ? "#27222e" : "white"};
  }
  
  #blank-grid {
    border: ${(props) =>
      props.theme.mode === "dark" ? "2px solid white" : "2px solid black"};
  }

  .grid-status-display, .grid-status-display-yellow, 
  .grid-status-display-green, .grid-status-display-gray {
    color: ${(props) => (props.theme.mode === "dark" ? "#f6f6ff" : "#121212")};
    border: ${(props) =>
      props.theme.mode === "dark" ? "2px solid white" : "2px solid black"};
  }

  .layer-outer-button, .layer-info-button {
    color: ${(props) => (props.theme.mode === "dark" ? "#f6f6ff" : "#121212")};
  }

  .layer-choice, .layer-info {
    background-color: ${(props) =>
      props.theme.mode === "dark" ? "var(--tertiary)" : "var(--add-layer)"};
    color: ${(props) =>
      props.theme.mode === "dark" ? "var(--add-layer)" : "var(--tertiary)"};
  }

  .response-container {
    background-color: ${(props) =>
      props.theme.mode === "dark" ? "#1e1e1e" : "var(--add-layer)"};
  }

  .dashboard-header, .row-style {
    color: ${(props) =>
      props.theme.mode === "dark" ? "#f6f6ff" : "#121212"} !important;
  }
`;

export default GlobalStyle;
