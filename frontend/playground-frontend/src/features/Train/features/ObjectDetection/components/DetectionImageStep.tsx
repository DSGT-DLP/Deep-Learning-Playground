import FilerobotImageEditor, {
  TABS,
  TOOLS,
} from "react-filerobot-image-editor";
import { useLazySendUploadDataQuery } from "../redux/uploadApi";

const DetectionImageStep = () => {
  const [sendUploadData, { data }] = useLazySendUploadDataQuery();
  const dataURLtoFile = (
    dataurl: string | undefined,
    filename: string | undefined
  ) => {
    const arr = dataurl.split(",");
    const mime = arr[0].match(/:(.*?);/)[1];
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
      onSave={(editedImageObject) => {
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
            ratio: 4 / 3,
          },
          {
            titleKey: "cinemascope",
            descriptionKey: "21:9",
            ratio: 21 / 9,
          },
        ],
        presetsFolders: [
          {
            titleKey: "socialMedia", // will be translated into Social Media as backend contains this translation key
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
    />
  );
};

export default DetectionImageStep;
