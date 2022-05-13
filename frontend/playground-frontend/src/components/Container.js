import { memo } from "react";
import LayerChoice2 from "./LayerChoice2.js";
import AddNewLayer2 from "./AddNewLayer2.js";

const Container = () => {
  return (
    <div>
      <AddNewLayer2 />
      <LayerChoice2 name="Glass" />
      <LayerChoice2 name="Banana" />
      <LayerChoice2 name="Paper" />
    </div>
  );
};

export default Container;
