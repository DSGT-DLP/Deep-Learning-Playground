import React, { useEffect, useState } from "react";
import Button from "react-bootstrap/Button";
import Form from "react-bootstrap/Form";
import { useAppDispatch, useAppSelector } from "@/common/redux/hooks";
import {
  isSignedIn,
  updateUserDisplayName,
  updateUserEmail,
  updateUserPassword,
} from "@/common/redux/userLogin";
import { useRouter } from "next/router";
import NavbarMain from "@/common/components/NavBarMain";
import Footer from "@/common/components/Footer";
import { toast } from "react-toastify";
import { SerializedError } from "@reduxjs/toolkit";

const SettingsBlock = () => {
  const [fullName, setFullName] = useState<string>("");
  const [email, setEmail] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [checkPassword, setCheckedPassword] = useState<string>("");
  const user = useAppSelector((state) => state.currentUser.user);
  const dispatch = useAppDispatch();
  if (!isSignedIn(user)) {
    return <></>;
  }
  return (
    <Form>
      <h2>View or Change your Account Settings </h2>
      <Form.Group className="mb-3" controlId="update-name">
        <Form.Label>Full Name</Form.Label>
        <Form.Control
          placeholder={user.displayName}
          onBlur={(e) => setFullName(e.target.value)}
          size="lg"
        />
      </Form.Group>
      <Form.Group className="mb-3" controlId="update-email">
        <Form.Label>Email address</Form.Label>
        <Form.Control
          type="email"
          placeholder={user.email}
          onBlur={(e) => setEmail(e.target.value)}
          autoComplete="email"
          size="lg"
        />
      </Form.Group>
      <Form.Group className="mb-3" controlId="update-password">
        <Form.Label>Password</Form.Label>
        <Form.Control
          type="password"
          placeholder={"New Password"}
          onBlur={(e) => setPassword(e.target.value)}
          aria-describedby="passwordHelpBlock"
          size="lg"
        />
      </Form.Group>
      <Form.Group className="mb-3" controlId="update-check-password">
        <Form.Label>Re-Type Password</Form.Label>
        <Form.Control
          type="password"
          placeholder={"New Password"}
          onBlur={(e) => setCheckedPassword(e.target.value)}
          size="lg"
        />
      </Form.Group>
      <div
        className="email-buttons d-flex flex-column"
        onClick={async () => {
          Promise.allSettled([
            (async () => {
              if (fullName) {
                try {
                  await dispatch(
                    updateUserDisplayName({ displayName: fullName })
                  ).unwrap();
                  toast.success("Successfully updated display name", {
                    position: toast.POSITION.TOP_CENTER,
                  });
                } catch (e) {
                  toast.error(
                    `Display name - ${(e as SerializedError).message}`,
                    {
                      position: toast.POSITION.TOP_CENTER,
                    }
                  );
                }
              }
            })(),
            (async () => {
              if (email) {
                try {
                  await dispatch(updateUserEmail({ email })).unwrap();
                  toast.success("Successfully updated email", {
                    position: toast.POSITION.TOP_CENTER,
                  });
                } catch (e) {
                  toast.error(`Email - ${(e as SerializedError).message}`, {
                    position: toast.POSITION.TOP_CENTER,
                  });
                }
              }
            })(),
            (async () => {
              if (password) {
                try {
                  await dispatch(
                    updateUserPassword({
                      password,
                      checkPassword,
                    })
                  ).unwrap();
                  toast.success("Successfully updated password", {
                    position: toast.POSITION.TOP_CENTER,
                  });
                } catch (e) {
                  toast.error(`Password - ${(e as SerializedError).message}`, {
                    position: toast.POSITION.TOP_CENTER,
                  });
                }
              }
            })(),
          ]);
        }}
      >
        <Button id="update-profile" className="mb-2">
          Update Profile
        </Button>
      </div>
    </Form>
  );
};

const AccountSettings = () => {
  const user = useAppSelector((state) => state.currentUser.user);
  const router = useRouter();
  useEffect(() => {
    if (!user) {
      router.replace("/login");
    }
  }),
    [user];
  if (!isSignedIn(user)) {
    return <></>;
  }
  return (
    <>
      <NavbarMain />
      <div id="accountSettings">
        <div id="header-section">
          <h1 className="headers">User Settings</h1>
        </div>
        <div
          className="sections"
          id="User Settings"
          data-testid="user-settings"
        >
          <SettingsBlock />
        </div>
      </div>
      <Footer />
    </>
  );
};

export default AccountSettings;
