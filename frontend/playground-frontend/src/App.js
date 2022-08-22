import React, { useState } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import {
  About,
  LoginPopup,
  Wiki,
  Feedback,
  Navbar,
  ImageModels,
  Footer,
  UserSettings,
} from "./components";
import { toast, ToastContainer } from "react-toastify";
import Home from "./Home";
import "react-toastify/dist/ReactToastify.css";
import "./App.css";
import PropTypes from "prop-types";
import { auth } from "./firebase";
import { useAuthState } from "react-firebase-hooks/auth";

const AuthRoute = ({ children }) => {
  const [user, loading] = useAuthState(auth);

  if (loading) return <></>;

  // check if authenticated
  if (user) {
    return children;
  } else {
    toast.error("User not logged in");
    return <Navigate to="/" replace />;
  }
};

function App() {
  const [showLogin, setShowLogin] = useState(false);

  return (
    <div id="app">
      <Router>
        <Navbar setShowLogin={setShowLogin} />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/img-models" element={<ImageModels />} />
          <Route path="/About" element={<About />} />
          <Route path="/Wiki" element={<Wiki />} />
          <Route
            path="/usersettings"
            element={
              <AuthRoute>
                <UserSettings />
              </AuthRoute>
            }
          />
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

AuthRoute.propTypes = {
  children: PropTypes.node.isRequired,
};
