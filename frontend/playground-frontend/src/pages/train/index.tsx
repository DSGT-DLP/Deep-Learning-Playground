import Footer from "@/common/components/Footer";
import NavbarMain from "@/common/components/NavBarMain";
import React, { useEffect } from "react";
import { useAppSelector } from "@/common/redux/hooks";
import { useRouter } from "next/router";
import { isSignedIn } from "@/common/redux/userLogin";
import CreateTrainspace from "@/features/Train/components/CreateTrainspace";

const Trainspace = () => {
  const user = useAppSelector((state) => state.currentUser.user);
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
      <CreateTrainspace />
      <Footer />
    </div>
  );
};

export default Trainspace;
