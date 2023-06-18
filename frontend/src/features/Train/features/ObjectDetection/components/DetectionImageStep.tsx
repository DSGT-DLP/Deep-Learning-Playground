import FilerobotImageEditor, {
  TABS,
  TOOLS,
} from "react-filerobot-image-editor";
import { useLazySendUploadDataQuery } from "../redux/uploadApi";
import {
  DatasetStepTabLayout,
  DefaultDatasetPanel,
  UploadDatasetPanel,
} from "@/features/Train/components/DatasetStepLayout";
import { useForm } from "react-hook-form";

const DetectionImageStep = ({
  renderStepperButtons,
  setIsModified,
}: {
  renderStepperButtons: (
    submitTrainspace: (data: TrainspaceData<"DATASET">) => void
  ) => React.ReactNode;
  setIsModified: React.Dispatch<React.SetStateAction<boolean>>;
}) => {
  const [sendUploadData, { data }] = useLazySendUploadDataQuery();
  const dataURLtoFile = (dataurl: string, filename: string) => {
    const arr = dataurl.split(",");
    if (arr.length === 0) {
      return new File([""], filename);
    }
    const matched = arr[0].match(/:(.*?);/);
    const mime = matched ? matched[1] : undefined;
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n) {
      u8arr[n - 1] = bstr.charCodeAt(n - 1);
      n -= 1; // to make eslint happy
    }
    return new File([u8arr], filename, { type: mime });
  };
  return (
    <FilerobotImageEditor
      source="https://scaleflex.airstore.io/demo/stephen-walker-unsplash.jpg"
      onSave={(editedImageObject: any) => {
        const file = dataURLtoFile(
          editedImageObject.imageBase64,
          editedImageObject.fullName
        );
        sendUploadData(file);
      }}
      annotationsCommon={{
        fill: "#ff0000",
      }}
      Text={{ text: "Filerobot..." }}
      Rotate={{ angle: 90, componentType: "slider" }}
      Crop={{
        presetsItems: [
          {
            titleKey: "classicTv",
            descriptionKey: "4:3",
            ratio: (4 / 3).toString(),
          },
          {
            titleKey: "cinemascope",
            descriptionKey: "21:9",
            ratio: (21 / 9).toString(),
          },
        ],
        presetsFolders: [
          {
            titleKey: "socialMedia",

            // icon: Social, // optional, Social is a React Function component. Possible (React Function component, string or HTML Element)
            groups: [
              {
                titleKey: "facebook",
                items: [
                  {
                    titleKey: "profile",
                    width: 180,
                    height: 180,
                    descriptionKey: "fbProfileSize",
                  },
                  {
                    titleKey: "coverPhoto",
                    width: 820,
                    height: 312,
                    descriptionKey: "fbCoverPhotoSize",
                  },
                ],
              },
            ],
          },
        ],
      }}
      tabsIds={[TABS.ADJUST, TABS.ANNOTATE, TABS.WATERMARK]} // or {['Adjust', 'Annotate', 'Watermark']}
      defaultTabId={TABS.ANNOTATE} // or 'Annotate'
      defaultToolId={TOOLS.TEXT} // or 'Text'
      savingPixelRatio={0}
      previewPixelRatio={0}
    />
  );
};

export default DetectionImageStep;
