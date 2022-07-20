import React, {useState} from "react";
import Transforms from "./Transforms";
import AddedLayer from "../Home/AddedLayer";
import { POSSIBLE_TRANSFORMS } from "../../settings";

const ImageModels = () => {
    const [trainTransforms, setTrainTransforms] = useState([]);
    const [testTransforms, setTestTransforms] = useState([]);

    const input_responses = {
        trainTransforms: trainTransforms,
        testTransforms: testTransforms,
    };

    return (
        <div>

            <Transforms 
                queryText = {"Train Transform"}
                options = {POSSIBLE_TRANSFORMS}
                transforms = {trainTransforms}
                setTransforms = {setTrainTransforms}
            />

            <Transforms
                queryText={"Test Transform"}
                options = {POSSIBLE_TRANSFORMS}
                transforms = {testTransforms}
                setTransforms = {setTestTransforms}
            />

        </div>
    )
}

export default ImageModels;