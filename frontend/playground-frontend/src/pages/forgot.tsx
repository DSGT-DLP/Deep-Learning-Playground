// @flow
import React, { useState } from "react";
import type { ReactNode } from "react";
import Button from "react-bootstrap/Button";
import Form from "react-bootstrap/Form";
import { sendPasswordResetEmail } from "firebase/auth";
import { toast } from "react-toastify";
import Link from "next/link";
import NavbarMain from "@/common/components/NavBarMain";
import { auth } from "@/common/utils/firebase";
import Footer from "@/common/components/Footer";

function Forgot() {
  const [email, setEmail] = useState("");

  const onChange = (e: React.ChangeEvent<HTMLInputElement>) =>
    setEmail(e.target.value);

  const onSubmit = async () => {
    try {
      await sendPasswordResetEmail(auth, email);
      toast.success("Email was sent");
    } catch (error) {
      toast.error("Could not send reset email");
    }
  };

  const EmailNewPassword: ReactNode = (
    <>
      <Form.Group className="mb-3" controlId="login-email">
        <h2 className="title">Forgot Password</h2>
        <Form.Label>Email address</Form.Label>
        <Form.Control
          type="email"
          placeholder="someone@example.com"
          value={email}
          onChange={onChange}
        />
      </Form.Group>

      <div className="d-flex flex-column">
        <Button
          id="forgot-password"
          className="buttons mb-2"
          onClick={onSubmit}
        >
          Send Reset Link
        </Button>
        <Link className="forgetPasswordlink" href="/login">
          Sign In
        </Link>
      </div>
    </>
  );

  return (
    <>
      <NavbarMain />
      <div id="forgot-page" className="text-center">
        <div className="main-container mt-5 mb-5">
          <Form className="form-container p-5">{EmailNewPassword}</Form>
        </div>
      </div>
      <Footer />
    </>
  );
}

export default Forgot;
