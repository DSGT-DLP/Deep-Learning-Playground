import { updateTrainspaceData } from "@/features/Train/redux/trainspaceSlice";
import { TrainspaceData } from "../types/imageTypes";
import { TRAINSPACE_SETTINGS } from "../constants/imageConstants";

export const updateImageTrainspaceData = <
  T extends (typeof TRAINSPACE_SETTINGS)["steps"][number]
>({
  current,
  stepLabel,
}: {
  current: TrainspaceData<T>;
  stepLabel: T;
}) =>
  updateTrainspaceData({
    ...current,
    ...{
      step:
        TRAINSPACE_SETTINGS.steps.findIndex((step) => step === stepLabel) + 1,
    },
  });
