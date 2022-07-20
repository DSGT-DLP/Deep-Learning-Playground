import Input from "../Home/Input";
import AddedLayer from "../Home/AddedLayer";
import React, {useState} from "react";


const Transforms = (props) => {
    const {queryText, options, transforms, setTransforms} = props;

    const addTransform = (e, transforms, setTransforms) => {
        const copyTransform = [...transforms];
        const selectedTransform = deepCopyObj(e);
        Object.values(selectedTransform.parameters).forEach((val) => {
            val["value"] = "";
        });
        copyTransform.push(selectedTransform);
        setTransforms(copyTransform);
    }

    return (
        <div>
            <div style={{display: "inline-block"}}>
            <Input
                queryText={queryText}
                options={options}
                onChange={(e) => {addTransform(
                    e, transforms, setTransforms)}
                }
                style={{float: "left"}}
            />

            {transforms.map((_, i) => (
            <div key={i} style={{display: "inline-block"}}>
                <AddedLayer
                    thisLayerIndex={i}
                    addedLayers={transforms}
                    setAddedLayers={setTransforms}
                    key={i}
                    onDelete={() => {
                        const currentLayers = [...transforms];
                        currentLayers.splice(i, 1);
                        setTransforms(currentLayers);
                    }}
                    />
                </div> 
            ))}
            </div>
        </div>
    );
};

export default Transforms;

const deepCopyObj = (obj) => JSON.parse(JSON.stringify(obj));