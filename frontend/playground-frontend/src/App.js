import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { About, Wiki, Feedback, Navbar } from "./components";
import Home from "./Home";
import "./App.css";
import Footer from "./components/Footer/Footer";

function App() {
  return (
    <div>
      <Router>
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/About" element={<About />} />
          <Route path="/Wiki" element={<Wiki />} />
          <Route path="/feedback" element={<Feedback />} />
        </Routes>
        <Footer/>
      </Router>
    </div>
  );
}
export default App;
