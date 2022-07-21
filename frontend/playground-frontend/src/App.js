import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { About, Wiki, Feedback, Navbar, Footer } from "./components";
import Home from "./Home";
import "./App.css";


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
