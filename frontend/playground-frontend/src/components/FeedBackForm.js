import EmailInput from "./EmailInput";
import React, { useState } from "react";
import { COLORS, GENERAL_STYLES } from "../constants";
import TitleText from "./mini_components/TitleText";
import ReCAPTCHA from "react-google-recaptcha";
export default function Feedback() {
  const [email, setEmail] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [feedback, setFeedback] = useState("");
  const [recaptcha, setRecaptcha] = useState("");
  const [submitted, setSubmitted] = useState(false);
  return (
    <div>
      <div id="header-section">
        <h1 className="header">Deep Learning Playground Feedback</h1>
      </div>

      <div
        className="sections"
        style={{
          // marginLeft: "3%",
          paddingTop: "1%",
          // paddingLeft: "2%",
          paddingBottom: "1%",
          backgroundColor: "#f6f6ff",
          // marginRight: "5%",
        }}
      >
        <h2>Feedback Form</h2>
        <p>
          {" "}
          Fill this feedback form for any bugs, feature requests or complains!
          We'll get back to as soon as we can.
        </p>
        <form>
          <TitleText text="First Name" />
          <input
            type="text"
            placeholder="John"
            onChange={(e) => setFirstName(e.target.value)}
          />
          {submitted == true && firstName.trim() == "" && recaptcha !== "" && (
            <p>First Name cannot be blank</p>
          )}
          <TitleText text="Last name " />
          <input
            type="text"
            placeholder="Doe"
            onChange={(e) => setLastName(e.target.value)}
          />
          {submitted == true && lastName.trim() == "" && recaptcha != "" && (
            <p>Last Name cannot be blank</p>
          )}
          <EmailInput setEmail={setEmail} />
          {submitted == true && email.trim() == "" && recaptcha !== "" && (
            <p>Email Cannot be blank</p>
          )}
          <TitleText text="Feedback" />
          <textarea
            placeholder="Type your feedback here"
            rows="15"
            cols="60"
            style={{
              borderRadius: "10px",
              borderWidth: "0.5px",
              padding: "5px",
            }}
            onChange={(e) => setFeedback(e.target.value)}
          />
          {submitted == true && feedback == "" && recaptcha != "" && (
            <p>Please enter some feedback</p>
          )}
        </form>
        <div style={{ marginTop: "2%" }} />
        <ReCAPTCHA
          sitekey={process.env.REACT_APP_SITE_KEY}
          onChange={(e) => {
            setRecaptcha(e);
          }}
        />
        {submitted == true && recaptcha == "" && (
          <p>Please Complete ReCAPTCHA</p>
        )}
        <button
          style={{
            backgroundColor: COLORS.dark_blue,
            border: "none",
            height: "100%",
            width: "10%",
            ...GENERAL_STYLES.p,
            fontSize: 25,
            color: "white",
            marginTop: "2%",
          }}
          onClick={() => {
            setSubmitted(true);
            if (
              firstName.trim() !== "" &&
              lastName.trim() !== "" &&
              email.trim() !== "" &&
              feedback.trim() !== ""
            ) {
              // TODO: Send Mail
            }
            console.log(recaptcha);
            console.log(firstName + lastName + email + feedback);
          }}
        >
          Submit
        </button>
      </div>
    </div>
  );
}
