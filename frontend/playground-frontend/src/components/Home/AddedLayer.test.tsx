import {render, screen} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom";
import AddedLayer from "./AddedLayer";
import React from "react";
describe("AddedLayer_function", () => {
  // Tests that the ondelete function is called when the delete button is clicked and the layer is successfully deleted. tags: [happy path]
  it("test_deletes_layer_successfully", () => {
    const onDeleteMock = jest.fn();
    render(
      <AddedLayer
        thisLayerIndex={0}
        addedLayers={[{ display_name: "Test Layer", object_name: "My Layer", parameters: {} }]}
        setAddedLayers={() => {}}
        onDelete={onDeleteMock}
      />
    );
    const deleteButton = screen.getByTestId("delete-layer-button");
    userEvent.click(deleteButton);
    expect(onDeleteMock).toHaveBeenCalled();
  });

  // Tests that the addedlayer component renders a single layer component with its display name and parameters. tags: [happy path]
  it("test_renders_single_layer_component", () => {
    const { getByText } = render(
      <AddedLayer
        thisLayerIndex={0}
        addedLayers={[{ display_name: "Test Layer", object_name: "My Layer", parameters: {} }]}
        setAddedLayers={() => {}}
        onDelete={() => {}}
      />
    );
    expect(getByText("Test Layer")).toBeInTheDocument();
  });

  // Tests that the ondelete function is called when the delete button is clicked and the last layer in the addedlayers array is successfully deleted. tags: [edge case]
  it("test_deletes_last_layer_in_added_layers_array", () => {
    const onDeleteMock = jest.fn();
    render(
      <AddedLayer
        thisLayerIndex={0}
        addedLayers={[
          { display_name: "Test Layer 1", object_name: "My Layer 1", parameters: {} },
          { display_name: "Test Layer 2", object_name: "My Layer 2", parameters: {} },
        ]}
        setAddedLayers={() => {}}
        onDelete={onDeleteMock}
      />
    );
    const deleteButton = screen.getByTestId("delete-layer-button");
    userEvent.click(deleteButton);
    expect(onDeleteMock).toHaveBeenCalled();
  });

  // Tests that the addedlayer component renders a layer with no parameters when the parameters object is empty. tags: [edge case]
  it("test_renders_layer_with_no_parameters", () => {
    render(
      <AddedLayer
        thisLayerIndex={0}
        addedLayers={[{ display_name: "Test Layer", object_name: "Test Layer 1", parameters: {} }]}
        setAddedLayers={() => {}}
        onDelete={() => {}}
      />
    );
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  });

  // Tests that the addedlayers state is updated when a parameter value is changed. tags: [general behavior]
  it("test_updates_added_layers_state", () => {
    const setAddedLayersMock = jest.fn();
    const { getByRole } = render(
      <AddedLayer
        thisLayerIndex={0}
        addedLayers={[
          {
            display_name: "Test Layer",
            object_name: "My Layer",
            parameters: { param1: 
                { 
                    index: 0, 
                    parameter_name: "My Parameter", 
                    value: "",
                    min: undefined,
                    max: undefined,
                    parameter_type: "text" 
                } 
            },
          },
        ]}
        setAddedLayers={setAddedLayersMock}
        onDelete={() => {}}
      />
    );
    const inputBox = getByRole("textbox");
    userEvent.type(inputBox,  "new value");
    expect(setAddedLayersMock).toHaveBeenCalled();
  });

  // Tests that the parameters object for each layer is converted into an array of parameter objects. tags: [general behavior]
  it("test_converts_parameters_object_to_array", () => {
    render(
      <AddedLayer
        thisLayerIndex={0}
        addedLayers={[
          {
            display_name: "Test Layer",
            object_name: "My Layer",
            parameters: {
              param1: { 
                parameter_name: "Param 1", value: "",
                min: undefined,
                max: undefined,
                parameter_type: "text", 
                index: 0 
              },
              param2: { 
                parameter_name: "Param 2", value: "",
                min: undefined,
                max: undefined,
                parameter_type: "text",
                index: 1  
              },
            },
          },
        ]}
        setAddedLayers={() => {}}
        onDelete={() => {}}
      />
    );
    const paramInputs = screen.getAllByRole("textbox");
    expect(paramInputs.length).toBe(2);
  });
});
