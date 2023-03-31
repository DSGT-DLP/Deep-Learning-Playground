import React from "react";
import Form from "react-bootstrap/Form";
import { LayerParameter, ModelLayer } from "../../settings";

interface _InputOutputPromptResponsePropTypes {
  param_key: string;
  allParamInputs: { [key: string]: LayerParameter };
  setAddedLayers: (
    func: (currentAddedLayers: ModelLayer[]) => ModelLayer[]
  ) => void;
  thisLayerIndex: number;
}
const _InputOutputPromptResponse = (
  props: _InputOutputPromptResponsePropTypes
) => {
  const { param_key, allParamInputs, setAddedLayers, thisLayerIndex } = props;
  const { parameter_name, value, min, max, parameter_type } =
    allParamInputs[param_key];

  return (
    <div className="layer-param-container d-flex justify-content-between align-items-center">
      <p className="param_name">{parameter_name}</p>
      {parameter_type === "boolean" ? (
        <Form>
          <Form.Select className="layer-param-input-box">
            <option>True</option>
            <option>False</option>
          </Form.Select>
        </Form>
      ) : (
        <input
          type={parameter_type}
          min={min}
          max={max}
          value={value}
          onChange={(e) =>
            // updates the addedLayers state with the current user input value of parameters
            setAddedLayers((currentAddedLayers) => {
              const copyCurrent = [...currentAddedLayers];
              const parameters = copyCurrent[thisLayerIndex].parameters;
              parameters[param_key].value = e.target.value;
              return copyCurrent;
            })
          }
          className="layer-param-input-box free-response"
        />
      )}
    </div>
  );
};

interface AddedLayerPropTypes {
  thisLayerIndex: number;
  addedLayers: ModelLayer[];
  setAddedLayers: (
    func: (currentAddedLayers: ModelLayer[]) => ModelLayer[]
  ) => void;
  onDelete: React.MouseEventHandler<HTMLButtonElement>;
}
const AddedLayer = (props: AddedLayerPropTypes) => {
  const { thisLayerIndex, addedLayers, setAddedLayers, onDelete } = props;
  const thisLayer = addedLayers[thisLayerIndex];
  const { display_name, parameters } = thisLayer;

  // converts the parameters object for each layer into an array of parameter objects
  const param_array: React.ReactNode[] = [];
  Object.keys(parameters).forEach((key) => {
    param_array.push(
      <_InputOutputPromptResponse
        key={key}
        param_key={key}
        allParamInputs={thisLayer.parameters}
        setAddedLayers={setAddedLayers}
        thisLayerIndex={thisLayerIndex}
      />
    );
  });

  return (
    <div className="added-layer-container">
      <div className="layer-box layer-container text-center d-flex justify-content-center align-items-center">
        <button className="delete-layer" onClick={onDelete}>
          ❌
        </button>
        {display_name}
      </div>
      {param_array.length ? (
        <div className="input-box">{param_array}</div>
      ) : null}
    </div>
  );
};

export default AddedLayer;
