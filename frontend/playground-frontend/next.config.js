/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    REACT_APP_CAPTCHA_SITE_KEY: process.env.REACT_APP_CAPTCHA_SITE_KEY,
    REACT_APP_FEEDBACK_EMAIL: process.env.REACT_APP_FEEDBACK_EMAIL,
  },
};

module.exports = nextConfig;
