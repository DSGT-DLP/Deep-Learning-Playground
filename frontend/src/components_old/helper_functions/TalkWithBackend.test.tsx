import axios from "axios";
import { auth } from "../../firebase";
import {
    getSignedUploadUrl, uploadToBackend
} from "./TalkWithBackend";

jest.mock('axios', () => ({
    post: jest.fn(() => Promise.resolve({ data: {} })),
}));
  
jest.mock("../../firebase", () => ({
  auth: {
    currentUser: {
      getIdToken: jest.fn(),
      uid: "myuid",
      email: "myemail@example.com",
      displayName: "myname",
    },
  },
}));

describe("Test uploadToBackend function", () => {
    it("should call axios.post with correct arguments", async () => {
        const data = { key1: "value1", key2: "value2" };
        const token = "mytoken";
        (auth.currentUser?.getIdToken as jest.Mock).mockResolvedValueOnce(token);
        const headers = {
          Authorization: `Bearer ${token}`,
        };
        await uploadToBackend(data);
        expect(axios.post).toHaveBeenCalledWith("/api/upload", data, { headers });
    });
});

describe("Test getSignedUploadUrl function", () => {
  it("should call fetch with correct arguments", async () => {
    const version = 1;
    const filename = "test.png";
    const file = new File(["hello"], "test.png", { type: "image/png" });
    const headers = {
      Authorization: "Bearer mytoken",
    };
    const token = "mytoken";
        (auth.currentUser?.getIdToken as jest.Mock).mockResolvedValueOnce(token);
    const formData = new FormData();
    formData.append("version", "1");
    formData.append("filename", "test.png");
    formData.append("file", file);
    const expectedResult = { url: "https://example.com" };
    jest.spyOn(global, "fetch").mockResolvedValueOnce({
      json: jest.fn().mockResolvedValueOnce(expectedResult),
    } as any);
    const result = await getSignedUploadUrl(version, filename, file);
    expect(fetch).toHaveBeenCalledWith("/api/getSignedUploadUrl", {
      method: "POST",
      body: formData,
      headers: headers,
    });
  });
});
