import React, { useState } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
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
import { ToastContainer } from "react-toastify";
import Home from "./Home";
import "react-toastify/dist/ReactToastify.css";
import "./App.css";
import { useAuthState } from "react-firebase-hooks/auth";
import { auth } from "./firebase";

function App() {
  const [user, loading] = useAuthState(auth);
  const [showLogin, setShowLogin] = useState(false);
  const currentURL = window.location.href.split("/");
  const isOnLoginPage = currentURL[currentURL.length - 1] === "login";
  if (loading) {
    return <div id="app"></div>;
  }

  return (
    <div id="app">
      <BrowserRouter>
        {isOnLoginPage || <Navbar setShowLogin={setShowLogin} />}
        <Routes>
          <Route exact path="/" element={user ? <Dashboard /> : <Login />} />
          <Route path="/train" element={<Home />} />
          <Route path="/img-models" element={<ImageModels />} />
          <Route path="/about" element={<About />} />
          <Route path="/wiki" element={<Wiki />} />
          <Route path="/feedback" element={<Feedback />} />
        </Routes>
        <ToastContainer position="top-center" />
        {isOnLoginPage || <Footer />}
      </BrowserRouter>
      {showLogin && <LoginPopup setShowLogin={setShowLogin} />}
    </div>
  );
}
export default App;
