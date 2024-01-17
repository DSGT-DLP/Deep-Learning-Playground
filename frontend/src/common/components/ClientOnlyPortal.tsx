import { useRef, useEffect, useState, createPortal } from "react";
import { createPortal } from "react-dom";

export default function ClientOnlyPortal({
  children,
  selector,
}: {
  children: React.ReactNode;
  selector: string;
}) {
  const ref = useRef<Element>();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const portal = document.querySelector(selector);
    if (portal) {
      ref.current = portal;
      setMounted(true);
    }
  }, [selector]);

  return mounted && ref.current ? createPortal(children, ref.current) : null;
}
