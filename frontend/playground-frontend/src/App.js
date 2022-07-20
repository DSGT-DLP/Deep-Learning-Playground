import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { About, Login, Wiki, Feedback, Navbar } from "./components";
import Home from "./Home";
import "./App.css";

function App() {
  return (
    <div className="wrapper">
      <Router>
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/About" element={<About />} />
          <Route path="/login" element={<Login />} />
          <Route path="/Wiki" element={<Wiki />} />
          <Route path="/feedback" element={<Feedback />} />
        </Routes>
      </Router>
    </div>
  );
}
export default App;
