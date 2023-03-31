import React, { useState } from "react";
import { COLORS, LAYOUT } from "../../constants";
import { DropDown } from "..";
// import storage from 'local-storage-fallback';

interface InputPropType {
  queryText: string;
  options: object[];
  onChange: (e: unknown) => void;
  defaultValue: unknown;
  isMultiSelect: boolean;
  range: boolean;
  beginnerMode: boolean;
  freeInputCustomRestrictions: { type: string; min: number };
}
const Input = (props: InputPropType) => {
  const {
    queryText,
    range,
    options,
    onChange,
    defaultValue,
    freeInputCustomRestrictions,
    isMultiSelect,
    beginnerMode,
  } = props;

  const [numberinput, setNumberInput] = useState(0.2);
  const [rangeinput, setRangeInput] = useState(20);

  const changeNumber = (value: string) => {
    setNumberInput((Number(parseInt(value) / 100) * 20) / 20);
    setRangeInput(parseInt(value));
    onChange(Number(parseInt(value)) / 100);
  };
  const changeRange = (value: string) => {
    setNumberInput(Number(parseInt(value)));
    setRangeInput(Number(parseInt(value)) * 100);
    onChange(Number(parseInt(value)));
  };

  return (
    <div
      style={{
        flexDirection: LAYOUT.row.flexDirection,
        margin: 7.5,
        display: beginnerMode ? "none" : "flex",
      }}
    >
      <div style={styles.queryContainer}>
        <p className="queryText">{queryText}</p>
      </div>
      <div className="response-container">
        {options ? (
          <DropDown
            options={options}
            onChange={onChange}
            defaultValue={defaultValue}
            isMulti={isMultiSelect}
          />
        ) : (
          <>
            {range ? (
              <>
                <input
                  placeholder="Type..."
                  className="inputText"
                  type="number"
                  value={Number(numberinput)}
                  onChange={(e) => changeRange(e.target.value)}
                />
                <input
                  className="inputText"
                  type="range"
                  value={Number(rangeinput)}
                  onChange={(e) => changeNumber(e.target.value)}
                />
              </>
            ) : (
              <input
                className="inputText"
                placeholder="Type..."
                maxLength={64}
                {...freeInputCustomRestrictions}
                defaultValue={
                  defaultValue as
                    | string
                    | number
                    | readonly string[]
                    | undefined
                }
                onChange={(e) => {
                  if (freeInputCustomRestrictions?.type === "number")
                    onChange(Number(e.target.value));
                  else onChange(e.target.value);
                }}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default Input;

const styles = {
  queryContainer: {
    height: 50,
    width: 145,
    backgroundColor: COLORS.layer,
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  },
};
