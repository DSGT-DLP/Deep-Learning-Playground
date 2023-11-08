import React from "react";
import { URLs } from "../constants";

const Beginner = () => {
    return (
        <div>
            <h1>Michelle's Dummy Button</h1>
            <button
                style={{
                    backgroundColor: "#FFB9EF",
                    padding: "15px 32px",
                    textAlign: "center",
                    fontSize: "16px",
                }}
                disabled
            >
                Button
            </button>
        </div>
    );
};

export default Beginner;