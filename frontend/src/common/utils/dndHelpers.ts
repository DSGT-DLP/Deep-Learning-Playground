import {
  KeyboardSensor,
  KeyboardSensorOptions,
  PointerSensor,
  PointerSensorOptions,
  useSensor,
} from "@dnd-kit/core";

function shouldHandleEvent(element: HTMLElement | null) {
  let cur = element;

  while (cur) {
    if (cur.dataset && cur.dataset.noDnd) {
      return false;
    }
    cur = cur.parentElement;
  }

  return true;
}

class CustomPointerSensor extends PointerSensor {
  static activators = [
    {
      eventName: "onPointerDown" as const,
      handler: ({ nativeEvent: event }: React.MouseEvent) => {
        return shouldHandleEvent(event.target as HTMLElement);
      },
    },
  ];
}

class CustomKeyboardSensor extends KeyboardSensor {
  static activators = [
    {
      eventName: "onKeyDown" as const,
      handler: ({ nativeEvent: event }: React.KeyboardEvent<Element>) => {
        return shouldHandleEvent(event.target as HTMLElement);
      },
    },
  ];
}

/**
 * PointerSensor that ignores elements and its children with data-no-dnd attribute
 * @param options
 * @returns custom dnd-kit PointerSensor
 */
export const useCustomPointerSensor = (options?: PointerSensorOptions) =>
  useSensor(CustomPointerSensor, options);

/**
 * KeyboardSensor that ignores elements and its children with data-no-dnd attribute
 * @param options
 * @returns custom dnd-kit KeyboardSensor
 */
export const useCustomKeyboardSensor = (options?: KeyboardSensorOptions) =>
  useSensor(CustomKeyboardSensor, options);
