import "bootstrap/dist/css/bootstrap.min.css";
import "@/styles/globals.css";
import type { AppProps } from "next/app";
import Head from "next/head";
import React from "react";

export default function App({ Component, pageProps }: AppProps) {
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
      <Component {...pageProps} />
    </>
  );
}
