import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import {
  About,
  Wiki,
  Feedback,
  NavbarMain,
  ImageModels,
  Footer,
  Dashboard,
  Login,
} from "./components";
import Home from "./Home";
import { Navigate } from "react-router-dom";
import { ToastContainer } from "react-toastify";
import { useSelector } from "react-redux";

import "react-toastify/dist/ReactToastify.css";
import "./App.css";

function App() {
  const userEmail = useSelector((state) => state.currentUser.email);

  return (
    <div id="app">
      <BrowserRouter>
        <div id="app-router">
          <NavbarMain />
          <Routes>
            <Route
              exact
              path="/"
              element={
                userEmail ? (
                  <Navigate to="/dashboard" />
                ) : (
                  <Navigate to="/login" />
                )
              }
            />
            <Route
              path="/login"
              element={userEmail ? <Navigate to="/dashboard" /> : <Login />}
            />
            <Route path="/train" element={<Home />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/img-models" element={<ImageModels />} />
            <Route path="/about" element={<About />} />
            <Route path="/wiki" element={<Wiki />} />
            <Route path="/feedback" element={<Feedback />} />
          </Routes>
          <ToastContainer position="top-center" />
        </div>
        <Footer />
      </BrowserRouter>
    </div>
  );
}
export default App;
