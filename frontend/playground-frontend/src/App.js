import React, { useEffect, useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import {
  About,
  LoginPopup,
  Wiki,
  Feedback,
  Navbar,
  ImageModels,
  Footer,
} from "./components";
import { ToastContainer } from "react-toastify";
import Home from "./Home";
import "react-toastify/dist/ReactToastify.css";
import "./App.css";
import { updateUserSettings } from "./components/helper_functions/TalkWithBackend";

function App() {
  const [showLogin, setShowLogin] = useState(false);
  useEffect(() => {
    setTimeout(() => {
      updateUserSettings();
    }, 1000);
  });

  return (
    <div id="app">
      <Router>
        <Navbar setShowLogin={setShowLogin} />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/img-models" element={<ImageModels />} />
          <Route path="/About" element={<About />} />
          <Route path="/Wiki" element={<Wiki />} />
          <Route path="/feedback" element={<Feedback />} />
        </Routes>
        <ToastContainer position="top-center" />
        <Footer />
      </Router>
      {showLogin && <LoginPopup setShowLogin={setShowLogin} />}
    </div>
  );
}
export default App;
