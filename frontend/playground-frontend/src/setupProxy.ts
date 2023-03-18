import { createProxyMiddleware } from "http-proxy-middleware";

module.exports = (app: unknown) => {
  //eslint-disable-next-line @typescript-eslint/no-explicit-any
  (app as any).use(
    "/api",
    createProxyMiddleware({
      target: "http://127.0.0.1:8000",
    })
  );
};
