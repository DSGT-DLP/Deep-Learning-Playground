interface SpacerProps {
  height?: number;
  width?: number;
}

const Spacer = (props: SpacerProps) => {
  return <div style={{ height: props.height || 1, width: props.width || 1 }} />;
};

export default Spacer;
