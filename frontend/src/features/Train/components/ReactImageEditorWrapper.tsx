import ReactImageEditor from "@toast-ui/react-image-editor";
import React from "react";

export default function ReactImageEditorWrapper({ editorRef, ...props }) {
  return <ReactImageEditor {...props} ref={editorRef} />;
}
