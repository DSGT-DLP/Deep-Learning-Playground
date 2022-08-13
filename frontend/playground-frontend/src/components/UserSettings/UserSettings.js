/* eslint-disable no-unused-vars */
import React from "react";
import { ErrorMessage, Field, Form, Formik } from "formik";
import * as Yup from "yup";
import { auth } from "../../firebase";
import {
  useUpdateProfile,
  useUpdateEmail,
  useUpdatePassword,
} from "react-firebase-hooks/auth";

const UserSettings = () => {
  const [updateProfile, updatingProfile, updateProfileError] =
    useUpdateProfile(auth);
  const [updateEmail, updatingEmail, updateEmailError] = useUpdateEmail(auth);
  const [updatePassword, updatingPassword, updatePasswordError] =
    useUpdatePassword(auth);

  return (
    <div>
      <h1>Hi</h1>
      <Formik
        initialValues={{ email: "email" }}
        validationSchema={Yup.object({
          email: Yup.string("Email must be a valid string")
            .email("Email is invalid")
            .required("Email is required"),
        })}
        onSubmit={async (values, { setSubmitting }) => {}}
      >
        {(formik) => (
          <Form>
            <Field type="email" name="email" />
            <ErrorMessage name="email" />
            <button type="submit" disabled={formik.isSubmitting}>
              Submit
            </button>
          </Form>
        )}
      </Formik>
    </div>
  );
};

export default UserSettings;
