import React, { useState } from "react";
import {
  ChoiceTab,
  TitleText,
  RectContainer,
  BackgroundLayout,
  Input,
} from "..";
import { FaCloudUploadAlt } from "react-icons/fa";

const NLP = () => {
  const [nlpData, nlpDataChange] = useState("");
  const [vectorSize, setVectorSize] = useState(100);
  const [windowSize, setWindowSize] = useState(5);
  const [minCount, setMinCount] = useState(0);
  const [workers, setWorkers] = useState(4);
  const [blobUrl, setBlobUrl] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [inputWord, setInputWord] = useState("");
  const [wordSimilarities, setWordSimilarities] = useState("");

  const input_queries = [
    {
      queryText: "Vector Size",
      freeInputCustomRestrictions: { type: "number", min: 0 },
      onChange: setVectorSize,
      defaultValue: vectorSize,
    },
    {
      queryText: "Window Size",
      freeInputCustomRestrictions: { type: "number", min: 0 },
      onChange: setWindowSize,
      defaultValue: windowSize,
    },
    {
      queryText: "Min Count",
      freeInputCustomRestrictions: { type: "number", min: 0 },
      onChange: setMinCount,
      defaultValue: minCount,
    },
    {
      queryText: "Workers",
      freeInputCustomRestrictions: { type: "number", min: 0 },
      onChange: setWorkers,
      defaultValue: workers,
    },
  ];

  const handleNlpDataChange = (event) => {
    // 👇️ access textarea value
    nlpDataChange(event.target.value);
  };

  const handleInputWordChange = (event) => {
    setInputWord(event.target.value);
  };

  let output;

  const b64toBlob = (b64Data, contentType = "", sliceSize = 512) => {
    const byteCharacters = atob(b64Data);
    const byteArrays = [];

    for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
      const slice = byteCharacters.slice(offset, offset + sliceSize);

      const byteNumbers = new Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        byteNumbers[i] = slice.charCodeAt(i);
      }

      const byteArray = new Uint8Array(byteNumbers);
      byteArrays.push(byteArray);
    }

    const blob = new Blob(byteArrays, { type: contentType });
    return blob;
  };

  const showFile = async (e) => {
    e.preventDefault();
    const reader = new FileReader();
    reader.onload = async (e) => {
      const text = e.target.result;
      nlpDataChange(text);
    };
    reader.readAsText(e.target.files[0]);
  };

  const onClick = async () => {
    async function postData(url = "", data = {}) {
      // Default options are marked with *
      const response = await fetch(url, {
        method: "POST", // *GET, POST, PUT, DELETE, etc.
        //mode: 'no-cors', // no-cors, *cors, same-origin
        //cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
        //credentials: 'same-origin', // include, *same-origin, omit
        headers: {
          "Content-Type": "application/json",
          // 'Content-Type': 'application/x-www-form-urlencoded',
          Accept: "application/json",
        },
        //redirect: 'follow', // manual, *follow, error
        //referrerPolicy: 'no-referrer', // no-referrer, *no-referrer-when-downgrade, origin, origin-when-cross-origin, same-origin, strict-origin, strict-origin-when-cross-origin, unsafe-url
        body: JSON.stringify(data), // body data type must match "Content-Type" header
      });
      return response.json(); // parses JSON response into native JavaScript objects
    }
    output = await postData("/wordVec", {
      raw_review: nlpData,
      inputWord: inputWord,
      vectorSize: vectorSize,
      windowSize: windowSize,
      minCount: minCount,
      workers: workers,
    });
    const blob = b64toBlob(output["base64Str"], "octet-stream");
    setBlobUrl(URL.createObjectURL(blob));
    setSubmitted(true);
    setWordSimilarities(output["text_output"]);
  };

  return (
    <>
      <div style={{ padding: 20 }}>
        <ChoiceTab></ChoiceTab>
        <TitleText text="Word2Vec" />
        <label htmlFor="file-upload" className="custom-file-upload">
          <FaCloudUploadAlt /> {"Choose text file"}
        </label>
        <input
          type="file"
          name="file"
          id="file-upload"
          accept=".txt"
          //onChange={handleFileUpload}
          onChange={(e) => showFile(e)}
          style={{ width: "100%" }}
        />

        <textarea
          placeholder="Input your NLP data here by copy and paste or upload a text file to the left"
          rows="15"
          cols="60"
          value={nlpData}
          onChange={handleNlpDataChange}
        />

        <textarea
          placeholder="Input test word for NLP"
          rows="2"
          cols="20"
          value={inputWord}
          onChange={handleInputWordChange}
        />

        <hr></hr>

        <BackgroundLayout>
          {input_queries.map((e) => (
            <Input {...e} key={e.queryText} />
          ))}
        </BackgroundLayout>

        <RectContainer
        >
          <button
          id="train-button"
          className="btn btn-primary"
            style={{
              cursor: "pointer",
            }}
            onClick={onClick}
          >
            Train!
          </button>
        </RectContainer>

        <hr></hr>

        <span
          style={{
            marginLeft: 8,
            visibility: submitted ? "visible" : "hidden",
          }}
        >
          <a
            onClick={() => {
              console.log(blobUrl);
            }}
            href={blobUrl}
            download="word2vec.model"
            id="download_csv_res"
          >
            📄 Download word2vec model
          </a>
        </span>

        <textarea
          rows="6"
          cols="60"
          value={wordSimilarities}
          style={{ visibility: submitted ? "visible" : "hidden" }}
        />
      </div>
    </>
  );
};

export default NLP;
