import EmailInput from "../Home/EmailInput";
import ReCAPTCHA from "react-google-recaptcha";
import React, { useState } from "react";
import TitleText from "../general/TitleText";
import { COLORS, GENERAL_STYLES } from "../../constants";

const Feedback = () => {
  const [email, setEmail] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [feedback, setFeedback] = useState("");
  const [recaptcha, setRecaptcha] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [successful, setSuccessful] = useState(false);

  const onClickSubmit = () => {
    setSubmitted(true);
    if (
      firstName.trim() &&
      lastName.trim() &&
      email.trim() &&
      feedback.trim()
    ) {
      setSuccessful(send_feedback_mail(firstName, lastName, email, feedback));
    }
  };

  if (successful) {
    return (
      <>
        <div id="header-section">
          <h1 className="header">Deep Learning Playground Feedback</h1>
        </div>

        <div className="sections" style={styles.content_section}>
          <p>Feedback submitted!</p>
        </div>
      </>
    );
  }

  return (
    <>
      <div id="header-section">
        <h1 className="header">Deep Learning Playground Feedback</h1>
      </div>

      <div className="sections" style={styles.content_section}>
        <h2>Feedback Form</h2>
        <p>
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
          {submitted && firstName.trim() === "" && recaptcha !== "" && (
            <p style={GENERAL_STYLES.error_text}>First Name cannot be blank</p>
          )}

          <TitleText text="Last name" />
          <input
            type="text"
            placeholder="Doe"
            onChange={(e) => setLastName(e.target.value)}
          />
          {submitted && lastName.trim() === "" && recaptcha !== "" && (
            <p style={GENERAL_STYLES.error_text}>Last Name cannot be blank</p>
          )}

          <TitleText text="Email" />
          <EmailInput setEmail={setEmail} />
          {submitted && email.trim() === "" && recaptcha !== "" && (
            <p style={GENERAL_STYLES.error_text}>Email Cannot be blank</p>
          )}

          <TitleText text="Feedback" />
          <textarea
            placeholder="Type your feedback here"
            rows="15"
            cols="60"
            style={styles.feedback_area}
            onChange={(e) => setFeedback(e.target.value)}
          />
          {submitted && feedback === "" && recaptcha !== "" && (
            <p style={GENERAL_STYLES.error_text}>Please enter some feedback</p>
          )}
        </form>

        <div style={{ marginTop: "2%" }} />

        <ReCAPTCHA
          sitekey={process.env.REACT_APP_SITE_KEY}
          onChange={(e) => setRecaptcha(e)}
        />
        {submitted && recaptcha === "" && (
          <p style={GENERAL_STYLES.error_text}>Please Complete ReCAPTCHA</p>
        )}

        <button style={styles.submit_button} onClick={onClickSubmit}>
          Submit
        </button>
      </div>
    </>
  );
};

const send_feedback_mail = async (firstName, lastName, email, feedback) => {
  const runResult = await fetch("/sendemail", {
    method: "POST",
    body: JSON.stringify({
      email_address: process.env.REACT_APP_EMAIL,
      subject: "FEEDBACK - " + firstName + " " + lastName + " " + email,
      body_text: feedback,
    }),
  });
  const resultJson = await runResult.json();
  return resultJson.success;
};

export default Feedback;

const styles = {
  submit_button: {
    ...GENERAL_STYLES.p,
    backgroundColor: COLORS.dark_blue,
    border: "none",
    color: "white",
    cursor: "pointer",
    fontSize: 25,
    marginTop: "2%",
    padding: 8,
  },
  feedback_area: {
    borderRadius: "10px",
    borderWidth: "0.5px",
    padding: "5px",
  },
  content_section: {
    backgroundColor: COLORS.background,
    paddingBottom: "1%",
    paddingTop: "1%",
  },
};
