/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    REACT_APP_CAPTCHA_SITE_KEY: process.env.REACT_APP_CAPTCHA_SITE_KEY,
    REACT_APP_FEEDBACK_EMAIL: process.env.REACT_APP_FEEDBACK_EMAIL,
  },
  redirects: async () => {
    return [
      {
        source: "/",
        destination: "/login",
        permanent: true,
      },
    ];
  },
  rewrites: () => [
    {
      source: "/api/:path*",
      destination: "http://127.0.0.1:8000/api/:path*",
    },
  ],
};

module.exports = nextConfig;
