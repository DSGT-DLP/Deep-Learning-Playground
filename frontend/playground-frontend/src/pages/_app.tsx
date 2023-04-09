import "bootstrap/dist/css/bootstrap.min.css";
import "react-toastify/dist/ReactToastify.css";
import "@/common/styles/globals.css";
import type { AppProps } from "next/app";
import Head from "next/head";
import React, { useEffect } from "react";
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { Provider } from "react-redux";
import { setCurrentUser } from "@/common/redux/userLogin";
import { auth } from "@/common/utils/firebase";
import store from "@/common/redux/store";
import storage from "local-storage-fallback";
import { useAppDispatch, useAppSelector } from "@/common/redux/hooks";
import { ToastContainer } from "react-toastify";
//import { wrapper } from "@/common/redux/store";

const FirebaseAuthState = () => {
  const dispatch = useAppDispatch();
  const user = useAppSelector((state) => state.currentUser.user);
  const [isExpectingUser, setIsExpectingUser] = React.useState(true);
  useEffect(() => {
    const expectUser = storage.getItem("expect-user");
    if (expectUser) {
      setTimeout(() => {
        setIsExpectingUser(false);
      }, 3000);
    } else {
      if (user == "pending") {
        dispatch(setCurrentUser(undefined));
      }
    }
    auth.onAuthStateChanged((firebaseUser) => {
      if (firebaseUser && firebaseUser.email && firebaseUser.displayName) {
        storage.setItem("expect-user", "true");
        dispatch(
          setCurrentUser({
            email: firebaseUser.email,
            uid: firebaseUser.uid,
            displayName: firebaseUser.displayName,
            emailVerified: firebaseUser.emailVerified,
          })
        );
      }
    });
  }, []);
  useEffect(() => {
    if (!isExpectingUser && user == "pending") {
      storage.removeItem("expect-user");
      dispatch(setCurrentUser(undefined));
    }
  }, [isExpectingUser, user]);
  return <></>;
};
const App = ({ Component, pageProps }: AppProps) => {
  return (
    <>
      <Head>
        <meta charSet="utf-8" />
        <link rel="icon" href="/dlp-logo.ico" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="theme-color" content="#000000" />
        <meta
          name="Deep Learning Playground"
          content="Play and experiment with machine/deep learning tools, provided by Data Science at Georgia Tech"
        />
        <meta
          name="author"
          content="See CODEOWNERS in https://github.com/karkir0003/Deep-Learning-Playground"
        />
        <link rel="apple-touch-icon" href="/dlp-logo.png" />
        <link rel="manifest" href="/manifest.json" />

        <title>Deep Learning Playground</title>
      </Head>
      <Provider store={store}>
        <FirebaseAuthState />
        <Component {...pageProps} />
        <ToastContainer />
      </Provider>
    </>
  );
};

export default App;
