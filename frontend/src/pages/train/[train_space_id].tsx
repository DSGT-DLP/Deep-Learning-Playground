import Footer from "@/common/components/Footer";
import NavbarMain from "@/common/components/NavBarMain";
import { useRouter } from "next/router";
import React from "react";

const TrainSpace = () => {
  const { train_space_id } = useRouter().query;
  return (
    <div style={{ height: "100vh" }}>
      <NavbarMain />
      <h1>{train_space_id}</h1>
      <Footer />
    </div>
  );
};

export default TrainSpace;
