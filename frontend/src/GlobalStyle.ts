import { createGlobalStyle } from "styled-components";

interface ThemeProps {
  theme: {
    mode: "dark" | "light";
  };
}

const GlobalStyle = createGlobalStyle`
  body {
    background-color: ${(props: ThemeProps) =>
      props.theme.mode === "dark" ? "#121212" : "#f6f6ff"};  
    color: ${(props: ThemeProps) =>
      props.theme.mode === "dark" ? "#f6f6ff" : "#121212"};
  }

  .sections {
    background-color: ${(props: ThemeProps) =>
      props.theme.mode === "dark" ? "#121212" : "#f6f6ff"};
  }

  p, .inputText {
    color: ${(props: ThemeProps) =>
      props.theme.mode === "dark" ? "#f6f6ff" : "#121212"};
  }

  #blank-grid p {
    color: ${(props: ThemeProps) =>
      props.theme.mode === "dark" ? "#f6f6ff" : "#121212"};
  }

  .input-container {
    background-color: ${(props: ThemeProps) =>
      props.theme.mode === "dark" ? "#5A5A5A" : "#var(--input)"};
  }

  .added-layer-container .input-box {
    background-color: ${(props: ThemeProps) =>
      props.theme.mode === "dark" ? "#27222e" : "white"};
  }
  
  #blank-grid {
    border: ${(props: ThemeProps) =>
      props.theme.mode === "dark" ? "2px solid white" : "2px solid black"};
  }

  .grid-status-display, .grid-status-display-yellow, 
  .grid-status-display-green, .grid-status-display-gray {
    color: ${(props: ThemeProps) =>
      props.theme.mode === "dark" ? "#f6f6ff" : "#121212"};
    border: ${(props: ThemeProps) =>
      props.theme.mode === "dark" ? "2px solid white" : "2px solid black"};
  }

  .layer-outer-button, .layer-info-button {
    color: ${(props: ThemeProps) =>
      props.theme.mode === "dark" ? "#f6f6ff" : "#121212"};
  }

  .layer-choice, .layer-info {
    background-color: ${(props: ThemeProps) =>
      props.theme.mode === "dark" ? "var(--tertiary)" : "var(--add-layer)"};
    color: ${(props: ThemeProps) =>
      props.theme.mode === "dark" ? "var(--add-layer)" : "var(--tertiary)"};
  }

  .response-container {
    background-color: ${(props: ThemeProps) =>
      props.theme.mode === "dark" ? "#1e1e1e" : "var(--add-layer)"};
  }

  .dashboard-header, .row-style {
    color: ${(props: ThemeProps) =>
      props.theme.mode === "dark" ? "#f6f6ff" : "#121212"} !important;
  }
`;

export default GlobalStyle;
