import { BrowserRouter as Router, Routes, Route} from "react-router-dom";
import Navbar from "./components/Navbar/Navbar";
import Home from "./Home";
import About from "./components/About";
function App () {
  return (
    <Router>
      <Navbar />
       <Routes>
        <Route path ="/" element ={<Home/>}/>
        <Route path = "/About" element = {<About/>} />
      </Routes>
    </Router>
  );
}
export default App;
