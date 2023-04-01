import React from "react";
import { Spinner } from "gestalt";
import "gestalt/dist/gestalt.css";

const LoadingPage = () => {
  return (
    <div className="loading">
      <Spinner show accessibilityLabel="Spinner" />
    </div>
  );
};

export default LoadingPage;
