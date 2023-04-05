module.exports = {
  presets: [
    [
      "next/babel",
      {
        "preset-env": { targets: { node: "current" } },
        "transform-runtime": { regenerator: true },
        "styled-jsx": {},
        "class-properties": {},
      },
    ],
  ],
  plugins: [],
};
