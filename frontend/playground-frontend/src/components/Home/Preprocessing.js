import TitleText from "../general/TitleText";
import CodeMirror from "@uiw/react-codemirror";
import { python } from "@codemirror/lang-python";
import { basicSetup } from "codemirror";
import Button from "react-bootstrap/Button";
import React, { useState } from "react";
import { userCodeEval } from "../helper_functions/TalkWithBackend";
import PropTypes from "prop-types";

const Preprocessing = (props) => {
  const { data, setData, setColumns, fileName } = props;
  const startingCode =
    "import pandas as pd\n\ndef preprocess(df): \n  # put your preprocessing code here!\n  return df";
  const [userCode, setUserCode] = useState(startingCode);
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
      <Button
        onClick={async () => {
          const response = await userCodeEval(data, userCode, fileName);
          setData(response["data"]);
          setColumns(response["columns"]);
        }}
      >
        Preprocess
      </Button>
    </div>
  );
};
Preprocessing.propTypes = {
  data: PropTypes.array.isRequired,
  setData: PropTypes.func.isRequired,
  setColumns: PropTypes.func.isRequired,
  fileName: PropTypes.any,
};

export default Preprocessing;
