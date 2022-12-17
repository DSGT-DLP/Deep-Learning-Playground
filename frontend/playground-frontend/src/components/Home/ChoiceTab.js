import React from "react";
import { COLORS } from "../../constants";
import { Link, useLocation } from "react-router-dom";

const ChoiceTab = () => {
  const loc = useLocation().pathname;

  const getBackground = (id) => {
    if (id === loc) return COLORS.gold;
    return COLORS.dark_blue;
  };

  return (
    <div>
      <button
        style={{ ...styles.button, backgroundColor: getBackground("/train") }}
      >
        <Link to="/train" style={styles.linkelEment}>
          Tabular Data
        </Link>
      </button>

      <button
        style={{
          ...styles.button,
          backgroundColor: getBackground("/img-models"),
        }}
      >
        <Link to="/img-models" style={styles.linkelEment}>
          Image Models
        </Link>
      </button>

      <button
        style={{
          ...styles.button,
          backgroundColor: getBackground("/classical-ml"),
        }}
      >
        <Link to="/classical-ml" style={styles.linkelEment}>
          Classical ML
        </Link>
      </button>
    </div>
  );
};

export default ChoiceTab;

const styles = {
  button: {
    marginRight: "5px",
    marginLeft: "5px",
    borderColor: COLORS.background,
    borderStyle: "solid",
    paddingLeft: "10px",
    paddingRight: "10px",
    borderRadius: "25px",
    fontSize: "17px",
    color: COLORS.background,
  },
  linkelEment: {
    color: COLORS.background,
    textDecoration: "none",
  },
};
