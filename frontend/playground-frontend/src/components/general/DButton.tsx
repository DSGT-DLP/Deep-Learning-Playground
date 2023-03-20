interface DButtonProps {
  onClick?: () => void;
  style?: React.CSSProperties;
  disabled?: boolean;
  className?: string;
  children?: string;
}

const DButton = (props: DButtonProps) => {
  const { onClick, style, disabled, className, children } = props;
  return (
    <button
      className={className || "btn btn-primary"}
      onClick={onClick}
      disabled={disabled}
      style={style}
    >
      {children}
    </button>
  );
};

export default DButton;
