import React from "react";
import { Form, Formik } from "formik";
import { useNavigate } from "react-router-dom";
import { TextField, ProtectedField } from "../form_components/formfields";

const Info = () => (
  <div id="login-info">
    <p>Information goes here</p>
  </div>
);

const FormBody = ({ formik, navigate }) => (
  <Form className="w-full">
    <TextField
      label="Email"
      name="email"
      type="text"
      placeholder="myemail@gmail.com"
      disabled={formik.isSubmitting}
    />
    <ProtectedField
      label="Password"
      name="password"
      type="password"
      placeholder="password"
      disabled={formik.isSubmitting}
    />
    <div className="grid grid-cols-2 gap-4 mt-8 mb-3">
      <button
        onClick={() => navigate("/")}
        disabled={formik.isSubmitting}
        type="button"
      >
        Cancel
      </button>
      <button disabled={formik.isSubmitting} type="submit">
        Log in
      </button>
    </div>
  </Form>
);

const Login = () => {
  const navigate = useNavigate();
  return (
    <div id="login-section">
      <Formik
        initialValues={{ email: "", password: "" }}
        validationSchema={undefined}
        onSubmit={(values, submitProps) => {}}
      >
        {(formik) => <FormBody formik={formik} navigate={navigate} />}
      </Formik>
    </div>
  );
};

const About = () => {
  return (
    <div id="login-page">
      <div id="login-main-section">
        <Info />
        <Login />
      </div>
      {/* <footer/> */}
    </div>
  );
};

export default About;
