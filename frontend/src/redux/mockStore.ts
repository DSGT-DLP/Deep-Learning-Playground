import configureMockStore from "redux-mock-store";

const mockStore = configureMockStore();

const mockedStore = mockStore({
  currentUser: {
    user: {
      email: "johndoe@gmail.com",
      uid: "12345",
      displayName: "John Doe",
      emailVerified: true,
    },
    userProgressData: null,
  },
  train: {
    customModelName: "John's Model",
    fileName: undefined,
    csvDataInput: [
      { col1: ["data1", "data2", "data3"] },
      { col2: ["data4", "data5", "data6"] },
    ],
    oldCsvDataInput: [
      { col1: ["data1", "data2", "data3"] },
      { col2: ["data4", "data5", "data6"] },
    ],
    uploadedColumns: ["col1", "col2"],
    fileURL: "data.csv",
  },
});

export default mockedStore;
