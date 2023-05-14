// @flow
import React, { useState } from "react";
import type { ReactNode } from "react";

import EmailInput from "@/common/components/EmailInput";
import TitleText from "@/common/components/TitleText";
import Spacer from "@/common/components/Spacer";
//import { EmailInput, TitleText, Spacer } from "../index";
import ReCAPTCHA from "react-google-recaptcha";

import { COLORS, GENERAL_STYLES } from "../constants";
import { toast } from "react-toastify";
import { InlineWidget } from "react-calendly";
import NavbarMain from "@/common/components/NavBarMain";
import { useLazySendFeedbackDataQuery } from "@/features/Feedback/redux/feedbackApi";

const CALENDLY_URL = "https://calendly.com/dlp-dsgt/30min";

function renderSuccessfulFeedbackSubmit(): ReactNode {
  return (
    <>
      <div id="header-section">
        <h1 className="header">Deep Learning Playground Feedback</h1>
      </div>

      <div className="sections" style={styles.content_section}>
        <p style={{ color: "0,0,0", fontSize: "5vh", textAlign: "center" }}>
          Feedback submitted!
        </p>
      </div>
      <div className="sections" style={styles.content_section}>
        <p style={{ color: "0,0,0", fontSize: "4vh", textAlign: "center" }}>
          If you would like to discuss your feedback, feel free to schedule a
          15-20 minute meeting over Calendly
        </p>
      </div>
      <div>
        <div>
          "
          <InlineWidget
            url={CALENDLY_URL}
            styles={{
              marginTop: "-66px",
              minWidth: "320px",
              height: "750px",
            }}
          />
        </div>
      </div>
    </>
  );
}

const Feedback = () => {
  const [email, setEmail] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [feedback, setFeedback] = useState("");
  const [recaptcha, setRecaptcha] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [successful, setSuccessful] = useState(false);
  const [sendFeedback, { data }] = useLazySendFeedbackDataQuery();
  const onClickSubmit = async () => {
    setSubmitted(true);
    if (
      firstName.trim() &&
      lastName.trim() &&
      email.trim() &&
      feedback.trim()
    ) {
      if (recaptcha != "") {
        const emailResult = await sendFeedback({
          email_address: process.env.REACT_APP_FEEDBACK_EMAIL,
          subject: "FEEDBACK - " + firstName + " " + lastName + " " + email,
          body_text: feedback,
        });
        if (!emailResult.isSuccess) {
          toast.error(emailResult.data.message);
        }
        console.log(emailResult);
        setSuccessful(emailResult.isSuccess);
      } else {
        toast.error("Please complete the ReCAPTCHA");
      }
    }
  };

  return (
    <>
      <NavbarMain />
      {successful ? (
        <>{renderSuccessfulFeedbackSubmit()}</>
      ) : (
        <>
          <div id="header-section">
            <h1 className="header">Deep Learning Playground Feedback</h1>
          </div>

          <div className="sections" style={styles.content_section}>
            <h2>Feedback Form</h2>
            <p>
              Fill this feedback form for any bugs, feature requests or
              complaints! We'll get back to as soon as we can.
            </p>

            <Spacer height={20} />

            <form>
              <TitleText text="First Name" />
              <input
                type="text"
                placeholder="John"
                onChange={(e) => setFirstName(e.target.value)}
              />
              {submitted && firstName.trim() === "" && recaptcha !== "" && (
                <p style={GENERAL_STYLES.error_text}>
                  First Name cannot be blank
                </p>
              )}

              <Spacer height={20} />

              <TitleText text="Last name" />
              <input
                type="text"
                placeholder="Doe"
                onChange={(e) => setLastName(e.target.value)}
              />
              {submitted && lastName.trim() === "" && recaptcha !== "" && (
                <p style={GENERAL_STYLES.error_text}>
                  Last Name cannot be blank
                </p>
              )}

              <Spacer height={20} />

              <TitleText text="Email" />
              <EmailInput email={email} setEmail={setEmail} />
              {submitted && email.trim() === "" && recaptcha !== "" && (
                <p style={GENERAL_STYLES.error_text}>Email Cannot be blank</p>
              )}

              <Spacer height={20} />

              <TitleText text="Feedback" />
              <textarea
                placeholder="Type your feedback here"
                style={styles.feedback_area}
                onChange={(e) => setFeedback(e.target.value)}
              />
              {submitted && feedback === "" && recaptcha !== "" && (
                <p style={GENERAL_STYLES.error_text}>
                  Please enter some feedback
                </p>
              )}
            </form>

            <div style={{ marginTop: "2%" }} />

            <ReCAPTCHA
              data-testid="recaptcha-feedback"
              sitekey={process.env.REACT_APP_CAPTCHA_SITE_KEY || ""}
              onChange={(e) => setRecaptcha(e || "")}
            />

            {submitted && recaptcha === "" && (
              <p style={GENERAL_STYLES.error_text}>Please Complete ReCAPTCHA</p>
            )}

            <button style={styles.submit_button} onClick={onClickSubmit}>
              Submit
            </button>
          </div>
        </>
      )}
    </>
  );
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
    width: "100%",
  },
  content_section: {
    backgroundColor: COLORS.background,
    paddingBottom: "1%",
    paddingTop: "1%",
  },
};
