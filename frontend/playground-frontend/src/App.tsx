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
  LearnMod,
  LearnContent,
  Forgot,
  AccountSettings,
  ObjectDetection,
} from "./components";
import Home from "./Home";
import { ToastContainer } from "react-toastify";
import React from "react";

import "react-toastify/dist/ReactToastify.css";
import "./App.css";
import { useAppSelector } from "./redux/hooks";

function App() {
  const user = useAppSelector((state) => state.currentUser.user);
  const verifyLogin = (target: JSX.Element) => (user ? target : <Login />);

  return (
    <div id="app">
      <BrowserRouter>
        <div id="app-router">
          <NavbarMain />
          <Routes>
            <Route
              path="/"
              element={
                user ? <Navigate to="/dashboard" /> : <Navigate to="/login" />
              }
            />
            <Route path="/login" element={<Login />} />
            <Route path="/train" element={verifyLogin(<Home />)} />
            <Route path="/img-models" element={verifyLogin(<ImageModels />)} />
            <Route
              path="/object-detection"
              element={verifyLogin(<ObjectDetection />)}
            />
            <Route
              path="/classical-ml"
              element={verifyLogin(<ClassicalMLModel />)}
            />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route
              path="/account-settings"
              element={verifyLogin(<AccountSettings />)}
            />
            <Route path="/about" element={<About />} />
            <Route path="/login" element={<Login />} />
            <Route path="/forgot" element={<Forgot />} />
            <Route path="/home" element={<Home />} />
            <Route path="/wiki" element={<Wiki />} />
            <Route path="/feedback" element={<Feedback />} />
            <Route path="/learn-mod" element={verifyLogin(<LearnMod />)} />
            <Route
              path="/LearnContent"
              element={verifyLogin(<LearnContent />)}
            />
          </Routes>
          <ToastContainer position="top-center" autoClose={2500} />
        </div>
        <Footer />
      </BrowserRouter>
    </div>
  );
}
export default App;
