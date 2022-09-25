import React, { useEffect, useState } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import {
  About,
  LoginPopup,
  Wiki,
  Feedback,
  Navbar,
  ImageModels,
  Footer,
  Dashboard,
  Login,
} from "./components";
import { onAuthStateChanged } from "firebase/auth";
import { auth } from "./firebase";
import { ToastContainer } from "react-toastify";
import Home from "./Home";

import "react-toastify/dist/ReactToastify.css";
import "./App.css";

function App() {
  return (
    <div id="app">
      <BrowserRouter>
        <Navbar />
        <Routes>
          <Route exact path="/" element={<Navigate to="/home" />} />
          <Route path="/login" element={<Login />} />
          <Route path="/home" element={<Home />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/img-models" element={<ImageModels />} />
          <Route path="/about" element={<About />} />
          <Route path="/wiki" element={<Wiki />} />
          <Route path="/feedback" element={<Feedback />} />
        </Routes>
        <ToastContainer position="top-center" />
        <Footer />
      </BrowserRouter>
    </div>
  );
}
export default App;
