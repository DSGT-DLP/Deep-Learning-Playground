import { React, useState } from "react";
import PropTypes from "prop-types";
import { COLORS, GENERAL_STYLES, LAYOUT } from "../../constants";
import { DropDown } from "..";
import storage from 'local-storage-fallback';

const Input = (props) => {
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

  const changeNumber = (userinput) => {
    setNumberInput((Number(userinput.target.value / 100) * 20) / 20);
    setRangeInput(userinput.target.value);
    onChange(Number(userinput.target.value) / 100);
  };
  const changeRange = (userinput) => {
    setNumberInput(Number(userinput.target.value));
    setRangeInput(Number(userinput.target.value) * 100);
    onChange(Number(userinput.target.value));
  };

  return (
    <div
      // @ts-ignore
      style={{
        ...LAYOUT.row,
        margin: 7.5,
        display: beginnerMode ? "none" : "flex",
      }}
    >
      <div style={styles.queryContainer}>
        <p
          // @ts-ignore
          style={styles.queryText}
        >
          {queryText}
        </p>
      </div>
      <div style={responseContainer()}>
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
                  style={styles.inputText}
                  type="number"
                  value={Number(numberinput)}
                  onChange={changeRange}
                />
                <input
                  style={styles.inputText}
                  type="range"
                  value={Number(rangeinput)}
                  onChange={changeNumber}
                />
              </>
            ) : (
              <input
                style={styles.inputText}
                placeholder="Type..."
                maxLength={64}
                {...freeInputCustomRestrictions}
                defaultValue={defaultValue}
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

Input.propTypes = {
  queryText: PropTypes.string.isRequired,
  options: PropTypes.arrayOf(PropTypes.object),
  onChange: PropTypes.func,
  freeInputCustomProps: PropTypes.object,
  defaultValue: PropTypes.oneOfType([
    PropTypes.object,
    PropTypes.number,
    PropTypes.string,
    PropTypes.array,
  ]),
  isMultiSelect: PropTypes.bool,
  range: PropTypes.bool,
  beginnerMode: PropTypes.bool,
  freeInputCustomRestrictions: PropTypes.shape({ type: PropTypes.string }),
  styles: PropTypes.array,
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
  queryText: {
    ...GENERAL_STYLES.p,
    color: "white",
    textAlign: "center",
    fontSize: 18,
    margin: 0,
  },
  responseText: {
    ...GENERAL_STYLES.p,
    color: "black",
    textAlign: "center",
    fontSize: 18,
  },
  responseDropDownButton: { border: "none", fontSize: 18, cursor: "pointer" },
  inputText: {
    ...GENERAL_STYLES.p,
    border: "none",
    backgroundColor: "transparent",
    width: "100%",
    textAlign: "center",
    fontSize: 18,
  },
};

function getInitialTheme () {
  const savedTheme = storage.getItem('theme');
  return savedTheme ? JSON.parse(savedTheme) : {mode: 'light'};
}

const theme = getInitialTheme();
const color = theme.mode === 'dark' ? "#27222e" : COLORS.addLayer;

function responseContainer () {
  return {
    height: 50,
    width: 170,
    backgroundColor: color,
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  };
}
