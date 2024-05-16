/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  trailingSlash: true,
  env: {
    REACT_APP_CAPTCHA_SITE_KEY: process.env.REACT_APP_CAPTCHA_SITE_KEY,
    REACT_APP_FEEDBACK_EMAIL: process.env.REACT_APP_FEEDBACK_EMAIL,
  },
  redirects: async () => {
    return [
      {
        source: "/",
        destination: "/login",
        permanent: false,
      },
    ];
  },
  rewrites: () => [
    {
      source: "/api/lambda/:path*",
      destination:
        "https://qt6nzp3sjd.execute-api.us-east-1.amazonaws.com/:path*",
    },
    {
      source: "/api/training/:path*",
      destination: "http://127.0.0.1:8000/api/:path*",
    },
  ],
};

module.exports = nextConfig;
