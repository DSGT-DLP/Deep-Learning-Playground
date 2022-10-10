import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import {
  About,
  Wiki,
  Feedback,
  NavbarMain,
  ImageModels,
  ClassicalMLModel,
  Footer,
  Dashboard,
  Login,
} from "./components";
import Home from "./Home";
import { ToastContainer } from "react-toastify";
import { getCookie } from "./components/helper_functions/Cookie";

import "react-toastify/dist/ReactToastify.css";
import "./App.css";

function App() {
  const userEmail = getCookie("userEmail");
  const verifyLogin = (target) => (userEmail ? target : <Login />);

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
            <Route path="/login" element={<Login />} />
            <Route path="/train" element={verifyLogin(<Home />)} />
            <Route path="/img-models" element={verifyLogin(<ImageModels />)} />
            <Route path="/classical-ml" element={verifyLogin(<ClassicalMLModel/>)}/>
            <Route path="/dashboard" element={<Dashboard />} />
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
