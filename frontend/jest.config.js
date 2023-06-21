module.exports = {
  preset: "ts-jest",
  transform: {
    "^.+\\.tsx?$": "babel-jest",
  },
  moduleDirectories: ['node_modules', 'public'],
};
