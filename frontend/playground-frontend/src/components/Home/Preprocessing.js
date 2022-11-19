import TitleText from "../general/TitleText";
import CodeMirror from "@uiw/react-codemirror";
import { python } from "@codemirror/lang-python";
import { basicSetup } from "codemirror";
import Button from "react-bootstrap/Button";
import React, { useState } from "react";
import { userCodeEval } from "../helper_functions/TalkWithBackend";

const Preprocessing = () => {
  const startingCode =
    "def preprocess(df): \n # put your preprocessing code here!";
  const [userCode, setUserCode] = useState("");
  const onChange = (value) => {
    setUserCode(value.target.innerText);
  };
  return (
    <div>
      <TitleText text="Preprocessing" />
      <CodeMirror
        value={startingCode}
        height="200px"
        extensions={[basicSetup, python()]}
        onBlur={onChange}
      />
      <Button onClick={() => userCodeEval(null, userCode)}>Preprocess</Button>
    </div>
  );
};

export default Preprocessing;
