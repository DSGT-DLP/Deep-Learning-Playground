/* eslint-disable no-unused-vars */
import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { useSelector } from "react-redux";
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
import { ToastContainer } from "react-toastify";

import "react-toastify/dist/ReactToastify.css";
import "./App.css";
import { getCookie } from "./components/helper_functions/Cookie";

function App() {
  const userEmailRedux = useSelector((state) => state.currentUser.email);
  const userEmail = getCookie("userEmail");
  const verifyLogin = (target) => userEmail ? target : <Navigate to="/login" />;
  const checkLogin = () => userEmail ? <Navigate to="/dashboard" />: <Login />;

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
            <Route path="/login" element={checkLogin()} />
            <Route path="/tabular-models" element={verifyLogin(<Home />)} />
            <Route path="/image-models" element={verifyLogin(<ImageModels />)} />
            <Route path="/dashboard" element={verifyLogin(<Dashboard />)} />
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
