import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import {
  About,
  LoginPopup,
  Wiki,
  Feedback,
  Navbar,
  ImageModels,
  Footer 
} from "./components";
import { ToastContainer } from "react-toastify";
import Home from "./Home";
import "react-toastify/dist/ReactToastify.css";
import "./App.css";
import {PopupProvider} from 'react-hook-popup';
function App() {
  const [showLogin, setShowLogin] = useState(false);

  return (
    
    <div id="app">
       <PopupProvider>
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
      </PopupProvider>
    </div>

  );
}
export default App;
