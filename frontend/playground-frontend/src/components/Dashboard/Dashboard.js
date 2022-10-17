import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from "@mui/material";
import React, { useState, useEffect } from "react";
import PropTypes from "prop-types";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import { useNavigate } from "react-router-dom";
import { getUserRecords } from "../helper_functions/TalkWithBackend";
import { useSelector } from "react-redux";

const BlankGrid = () => {
  const navigate = useNavigate();

  return (
    <div id="blank-grid-wrapper">
      <div id="blank-grid">
        <p>
          You haven't trained any models yet. Create your first model below!
        </p>
        <button id="blank-grid-button" onClick={() => navigate("/tabular-models")}>
          Train Model
        </button>
      </div>
    </div>
  );
};

const StatusDisplay = ({ status, progress }) => {
  if (status === "Queued") {
    return (
      <button className="grid-status-display grid-status-display-gray">
        Queued: {progress}
      </button>
    );
  } else if (status === "Uploading" || status === "Starting") {
    return (
      <button className="grid-status-display grid-status-display-blue">
        {status}
      </button>
    );
  } else if (status === "Training") {
    return (
      <button className="grid-status-display grid-status-display-yellow">
        Training: {progress}%
      </button>
    );
  } else if (status === "Success") {
    return (
      <button className="grid-status-display grid-status-display-green">
        Success <ArrowForwardIcon fontSize="small" />
      </button>
    );
  } else if (status === "Error") {
    return (
      <button className="grid-status-display grid-status-display-red">
        Error <ArrowForwardIcon fontSize="small" />
      </button>
    );
  } else {
    return <p>Incorrect status type passed</p>;
  }
};

const FilledGrid = ({ userRecords }) => {
  const navigate = useNavigate();

  return (
    <TableContainer style={{ display: "flex", justifyContent: "center" }}>
      <Table sx={{ minWidth: 400, m: 2 }}>
        <TableHead>
          <TableRow>
            <TableCell sx={{ fontWeight: "bold" }}>Name</TableCell>
            <TableCell sx={{ fontWeight: "bold" }}>Type</TableCell>
            <TableCell sx={{ fontWeight: "bold" }} align="left">
              Input File
            </TableCell>
            <TableCell sx={{ fontWeight: "bold" }} align="left">
              Date
            </TableCell>
            <TableCell sx={{ fontWeight: "bold" }} align="left">
              Status
            </TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {userRecords.map((row) => (
            <TableRow
              key={row.execution_id}
              sx={{
                "&:last-child td, &:last-child th": { border: 0 },
                cursor: "pointer",
              }}
              onClick={() => navigate("/tabular-models")}
              hover
            >
              <TableCell component="th" scope="row">
                {row.name}
              </TableCell>
              <TableCell component="th" scope="row">
                {row.data_source}
              </TableCell>
              <TableCell align="left">{row.file_name}</TableCell>
              <TableCell align="left">{row.timestamp}</TableCell>
              <TableCell align="left">
                <StatusDisplay
                  status={row.status}
                  progress={row.progress}
                />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

const Dashboard = () => {
  const [userRecords, setUserRecords] = useState([]);
  const userEmail = useSelector((state) => state.currentUser.email);

  useEffect(() => {
    if (userEmail) {
      (async function() {
        setUserRecords((await getUserRecords()).records);
      })();
    }
  }, [userEmail]);

  return (
    <div id="dashboard">
      <FilledGrid userRecords={userRecords} />
      {userRecords.length === 0 && <BlankGrid />}
    </div>
  );
};

export default Dashboard;

StatusDisplay.propTypes = {
  status: PropTypes.oneOf(["Queued", "Uploading", "Starting", "Training", "Success", "Error"]),
  progress: PropTypes.number,
};

FilledGrid.propTypes = {
  userRecords: PropTypes.array,
};
