import { useAppSelector } from "@/common/redux/hooks";
import { isSignedIn } from "@/common/redux/userLogin";
import CreateTrainspace from "@/features/Train/components/CreateTrainspace";
import { ALL_TRAINSPACE_SETTINGS } from "@/features/Train/constants/trainConstants";
import { BaseTrainspaceData } from "@/features/Train/types/trainTypes";
import { useRouter } from "next/router";
import { useEffect } from "react";

const Exercise = () => {
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
    <div>
      {trainspace ? (
        <TrainspaceInner trainspace={trainspace} />
      ) : (
        <CreateTrainspace />
      )}
    </div>
  );
};

const TrainspaceInner = ({
  trainspace,
}: {
  trainspace: BaseTrainspaceData;
}) => {
  const Component = ALL_TRAINSPACE_SETTINGS[trainspace.dataSource].component;
  return <Component />;
};

export default Exercise;
