import Footer from "@/common/components/Footer";
import NavbarMain from "@/common/components/NavBarMain";
import React, { useEffect } from "react";
import { useAppSelector } from "@/common/redux/hooks";
import { useRouter } from "next/router";
import { isSignedIn } from "@/common/redux/userLogin";
import CreateTrainspace from "@/features/Train/components/CreateTrainspace";
import TrainspaceSteps from "@/features/Train/components/TrainspaceSteps";

const Trainspace = () => {
  const user = useAppSelector((state) => state.currentUser.user);
  const trainspace = useAppSelector((state) => state.trainspace.current);
  const router = useRouter();
  useEffect(() => {
    if (router.isReady && !user) {
      router.replace({ pathname: "/login" });
    }
  }, [user, router.isReady]);
  if (!isSignedIn(user)) {
    return <></>;
  }
  return (
    <div style={{ height: "100vh" }}>
      <NavbarMain />
      {trainspace ? <TrainspaceSteps /> : <CreateTrainspace />}
      <Footer />
    </div>
  );
};

export default Trainspace;
