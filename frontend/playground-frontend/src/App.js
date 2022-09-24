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
import { useDispatch } from "react-redux";
import { setCurrentUser } from "./redux/userLogin";

import "react-toastify/dist/ReactToastify.css";
import "./App.css";

function App() {
  const [showLogin, setShowLogin] = useState(false);
  const dispatch = useDispatch();

  useEffect(() => {
    onAuthStateChanged(auth, (user) => {
      const userDetails = {
        email: user?.email ?? null,
        uid: user?.uid ?? null,
        displayName: user?.displayName ?? null,
        emailVerified: user?.emailVerified ?? null,
      };
      if (user) dispatch(setCurrentUser(userDetails));
    });
  }, []);

  return (
    <div id="app">
      <BrowserRouter>
        <Navbar setShowLogin={setShowLogin} />
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
      {showLogin && <LoginPopup setShowLogin={setShowLogin} />}
    </div>
  );
}
export default App;
