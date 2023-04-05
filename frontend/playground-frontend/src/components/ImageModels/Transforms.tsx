import Input from "../Home/Input";
import AddedLayer from "../Home/AddedLayer";
import React from "react";
import BackgroundLayout from "../Home/BackgroundLayout";
import { LayerParameter, ModelLayer, PossibleTransform } from "../../settings";

interface TransformsPropTypes {
  queryText: string;
  options: PossibleTransform[];
  transforms: ModelLayer[];
  setTransforms: React.Dispatch<React.SetStateAction<ModelLayer[]>>;
}
const Transforms = (props: TransformsPropTypes) => {
  const { queryText, options, transforms, setTransforms } = props;

  options.sort((a, b) => a.label.localeCompare(b.label));

  const addTransform = (
    e: ModelLayer,
    transforms: ModelLayer[],
    setTransforms: React.Dispatch<React.SetStateAction<ModelLayer[]>>
  ) => {
    const copyTransform = [...transforms];
    const selectedTransform = deepCopyObj(e);
    Object.values(selectedTransform.parameters).forEach((val) => {
      val.value = val.default
        ? val.default
        : (val as LayerParameter<"number" | "tuple">).min;
    });
    copyTransform.push(selectedTransform);
    setTransforms(copyTransform);
  };

  return (
    <div>
      <BackgroundLayout>
        <Input
          queryText={queryText}
          options={options}
          onChange={(e) => {
            addTransform(e as ModelLayer, transforms, setTransforms);
          }}
        />

        {transforms.map((_, i) => (
          <div key={i} style={{ display: "inline-block" }}>
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
      </BackgroundLayout>
    </div>
  );
};

export default Transforms;

const deepCopyObj = <T extends object>(obj: T): T =>
  JSON.parse(JSON.stringify(obj));
