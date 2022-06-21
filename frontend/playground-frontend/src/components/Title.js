// import DSGTLogo from "../images/logos/dsgt-logo-light.png";
import Navbar from "./Navbar/Navbar";
// import Footer from "./Footer/Footer";
import { COLORS, GENERAL_STYLES, LAYOUT } from "../constants";
const Title = () => {
  return (
    <div>
    <Navbar />
    {/* <Footer/> */}
    </div>

  );
};
export default Title;
const styles = {
  h1: {
    ...GENERAL_STYLES.p,
    padding:
      "100 0 0px 0" /* not working for some reason no longer on same level*/,
    margin: "0 0 0px 0",
    display: "flex",
    alignItems: "center",
    fontSize: 30,
    justifyContent: "space-between",
  },
  nav: { width: "67%" },
};
