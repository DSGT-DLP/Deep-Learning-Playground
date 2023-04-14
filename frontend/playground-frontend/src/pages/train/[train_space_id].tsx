import { useRouter } from "next/router";
import React from "react";

const TrainSpace = () => {
  const { train_space_id } = useRouter().query;
  return (
    <div>
      <h1>{train_space_id}</h1>
    </div>
  );
};

export default TrainSpace;
