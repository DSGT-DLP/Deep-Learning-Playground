import TitleText from "../general/TitleText";
import CodeMirror from "@uiw/react-codemirror";
import { toast } from "react-toastify";
import { python } from "@codemirror/lang-python";
import { basicSetup } from "codemirror";
import Button from "react-bootstrap/Button";
import React, { useState } from "react";
import { userCodeEval } from "../helper_functions/TalkWithBackend";
import { CSVInputDataColumnType, CSVInputDataRowType } from "./CSVInputFile";

interface PreprocessingPropTypes {
  data: CSVInputDataRowType[];
  setData: React.Dispatch<React.SetStateAction<CSVInputDataRowType[]>>;
  setColumns: React.Dispatch<React.SetStateAction<CSVInputDataColumnType[]>>;
}
const Preprocessing = (props: PreprocessingPropTypes) => {
  const { data, setData, setColumns } = props;
  const startingCode =
    "import pandas as pd\n\ndef preprocess(df): \n  # put your preprocessing code here!\n  return df";
  const [userCode, setUserCode] = useState(startingCode);
  return (
    <div>
      <TitleText text="Preprocessing" />
      <CodeMirror
        value={startingCode}
        height="200px"
        extensions={[basicSetup, python()]}
        onBlur={(e) => setUserCode(e.target.innerText)}
      />
      <Button
        onClick={async () => {
          const response = await userCodeEval(data, userCode);
          if (response["statusCode"] !== 200) {
            toast.error(response.message);
          } else {
            setData(response["data"]);

            const newColumns = response["columns"].map((c: string) => ({
              name: c,
              selector: (row: CSVInputDataRowType) => row[c],
            }));

            setColumns(newColumns);
            toast.success("Preprocessing successful!");
          }
        }}
      >
        Preprocess
      </Button>
    </div>
  );
};

export default Preprocessing;
