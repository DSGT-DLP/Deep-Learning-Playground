import { updateTrainspaceData } from "@/features/Train/redux/trainspaceSlice";
import { TrainspaceData } from "../types/tabularTypes";
import { TRAINSPACE_SETTINGS } from "../constants/tabularConstants";

export const updateTabularTrainspaceData = <
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
