import React, { useState } from "react";
import Transforms from "./Transforms";
import { POSSIBLE_TRANSFORMS } from "../../settings";
import { DEFAULT_ADDED_LAYERS, DEFAULT_TRANSFORMS } from "../../constants";

const ImageModels = () => {
  const [addedLayers, setAddedLayers] = useState(DEFAULT_ADDED_LAYERS);
  const [trainTransforms, setTrainTransforms] = useState(DEFAULT_TRANSFORMS);
  const [testTransforms, setTestTransforms] = useState(DEFAULT_TRANSFORMS);

  const input_responses = {
    trainTransforms: trainTransforms,
    testTransforms: testTransforms,
  };

  return (
    <div>
      <Transforms
        queryText={"Train Transform"}
        options={POSSIBLE_TRANSFORMS}
        transforms={trainTransforms}
        setTransforms={setTrainTransforms}
      />
      <Transforms
        queryText={"Test Transform"}
        options={POSSIBLE_TRANSFORMS}
        transforms={testTransforms}
        setTransforms={setTestTransforms}
      />
    </div>
  );
};

export default ImageModels;
