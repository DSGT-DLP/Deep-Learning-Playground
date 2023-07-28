import dynamic from "next/dynamic";
import { DATA_SOURCE } from "../types/trainTypes";
import React from "react";
import { Select, MenuItem, TextField, Button, Stack } from "@mui/material";
import { useUploadDatasetFileMutation } from "@/features/Train/redux/trainspaceApi";
import "tui-image-editor/dist/tui-image-editor.css";

const ReactImageEditorWrapper = dynamic(
  () => import("./ReactImageEditorWrapper"),
  {
    ssr: false,
  }
);

const ReactImageEditor = React.forwardRef((props, ref) => (
  <ReactImageEditorWrapper {...props} editorRef={ref} />
));

const dataURLtoFile = (dataurl: string, filename: string) => {
  const arr = dataurl.split(",");
  if (arr.length === 0) {
    return new File([""], filename);
  }
  const matched = arr[0].match(/:(.*?);/);
  const mime = matched ? matched[1] : undefined;
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n) {
    u8arr[n - 1] = bstr.charCodeAt(n - 1);
    n -= 1; // to make eslint happy
  }
  return new File([u8arr], filename, { type: mime });
};

const TuiImageEditor = ({ dataSource }: { dataSource: DATA_SOURCE }) => {
  const tui = React.useRef();
  const [fileName, setFileName] = React.useState("");
  const [fileType, setFileType] = React.useState("jpeg");
  const [uploadFile] = useUploadDatasetFileMutation();
  const uploadImage = (filename: string) => {
    const data = tui.current.getInstance().toDataURL({ fileType: fileType });
    const file = dataURLtoFile(data, filename);
    uploadFile({ dataSource, file });
  };
  return (
    <Stack alignItems="center" spacing={3}>
      <ReactImageEditor
        ref={tui}
        includeUI={{
          loadImage: {
            path: "",
          },
          menu: ["shape", "filter", "text", "mask", "icon", "draw", "crop"],
          initMenu: "",
          uiSize: {
            width: "1200px",
            height: "700px",
          },
          menuBarPosition: "bottom",
        }}
        selectionStyle={{
          cornerSize: 20,
          rotatingPointOffset: 70,
        }}
      />
      <Stack direction={"row"} spacing={2}>
        <TextField
          label="File Name"
          value={fileName}
          onChange={(e) => setFileName(e.target.value)}
        />
        <Select
          value={fileType}
          onChange={(e) => setFileType(e.target.value as string)}
        >
          <MenuItem value={"jpeg"}>jpeg</MenuItem>
          <MenuItem value={"png"}>png</MenuItem>
        </Select>
        <Button
          variant="outlined"
          onClick={() => uploadImage(fileName + "." + fileType)}
        >
          Upload
        </Button>
      </Stack>
    </Stack>
  );
};

export default TuiImageEditor;
