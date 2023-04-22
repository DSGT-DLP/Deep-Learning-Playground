import Footer from "@/common/components/Footer";
import NavbarMain from "@/common/components/NavBarMain";
import React, { useEffect } from "react";
import { useAppSelector } from "@/common/redux/hooks";
import { useRouter } from "next/router";
import { isSignedIn } from "@/common/redux/userLogin";
import CreateTrainspace from "@/features/Train/components/CreateTrainspace";
import { DATA_SOURCE_SETTINGS } from "@/features/Train/constants/trainConstants";

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
      {trainspace ? (
        <TrainspaceInner
          trainspaceComponent={
            DATA_SOURCE_SETTINGS[trainspace.dataSource].trainspaceComponent
          }
        />
      ) : (
        <CreateTrainspace />
      )}
      <Footer />
    </div>
  );
};

const TrainspaceInner = ({
  trainspaceComponent,
}: {
  trainspaceComponent: React.FC;
}) => {
  const TrainspaceComponent = trainspaceComponent;
  return <TrainspaceComponent />;
};

export default Trainspace;
