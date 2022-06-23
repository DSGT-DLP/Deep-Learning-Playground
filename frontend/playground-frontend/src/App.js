import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Title from "./components/Title";
import Home from "./Home";
import About from "./components/About/About";
import Wiki from "./components/Wiki/Wiki";
import "./App.css";
import ReactGA from "react-ga";
import RouteChangeTracker from "./components/RouteChangeTracker";

const TRACKING_ID = "UA-232631903-1";
ReactGA.initialize(TRACKING_ID, { testMode: process.env.NODE_ENV === "test" });

ReactGA.exception({
  description: "An error ocurred",
  fatal: true,
});

function App() {
  return (
    <div>
      <Router>
        <Title />
        <RouteChangeTracker />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/About" element={<About />} />
          <Route path="/Wiki" element={<Wiki />} />
        </Routes>
      </Router>
    </div>
  );
}
export default App;

